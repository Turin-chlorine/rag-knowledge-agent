"""Evidence-screen replacement CRUD cases, then freeze an uploadable v2 corpus.

Only publisher source texts are indexed. Candidate selection does not inspect
the RAG system's answers or scores. Every decision is retained in an audit log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from adjudicate import GOLD_SYSTEM, call_json, validate_gold
from evaluate import rows, write_json, write_jsonl
from prepare import digest, normalized, read_json, source_evidence_candidate


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "benchmarks"
V1 = BASE / "upload200" / "prepared"
V2 = BASE / "upload200_v2" / "prepared"
RAW = BASE / "crud" / "raw" / "split_merged.json"
REVIEW = BASE / "upload200" / "api_run" / "answerable180_reviews.jsonl"
REPAIR_ID = "crud-questanswer_1doc-64fa9b2cb82641eb8ecbf2a8"
REPAIRED_ANSWER = "美国编剧工会和演员工会分别在5月和7月开始全行业罢工。"


def candidates() -> list[tuple[str, dict, str]]:
    raw = read_json(RAW)
    used = {case["group_id"] for case in rows(V1 / "cases.jsonl")}
    result = []
    for n in (1, 2, 3):
        category = f"questanswer_{n}doc" if n == 1 else f"questanswer_{n}docs"
        pool = list(raw[category])
        random.Random(20260926 + n).shuffle(pool)
        for row in pool:
            case_id = f"crud-{category}-{row['ID']}"
            if str(row["ID"]) in used or not row.get("answers", "").strip() or any(not row.get(f"news{i}", "").strip() for i in range(1, n + 1)):
                continue
            split = "test" if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 4 == 0 else "dev"
            result.append((category, row, split))
    return result


def invalid_v1_cases() -> tuple[list[dict], dict[str, dict]]:
    original = rows(V1 / "cases.jsonl")
    review = {item["case_id"]: item for item in rows(REVIEW)}
    bad = [case for case in original if review.get(case["case_id"], {}).get("status") == "gold_not_supported_by_model_review"]
    if len(bad) != 13 or REPAIR_ID not in {case["case_id"] for case in bad}:
        raise ValueError("Unexpected v1 gold audit; do not silently change the replacement set")
    return bad, review


def screen() -> None:
    bad, _ = invalid_v1_cases()
    targets = Counter((case["category"], case["split"]) for case in bad if case["case_id"] != REPAIR_ID)
    selected_path = V2.parent / "replacement_selection.jsonl"
    audit_path = V2.parent / "candidate_audit.jsonl"
    selected = rows(selected_path) if selected_path.exists() else []
    audit = rows(audit_path) if audit_path.exists() else []
    prior = {item["case_id"] for item in audit}
    counts = Counter((item["category"], item["split"]) for item in selected)
    seen_events = {str(item["group_id"]) for item in selected}
    source_event_ids = {case["group_id"] for case in rows(V1 / "cases.jsonl")}
    for category, row, split in candidates():
        if counts[(category, split)] >= targets[(category, split)]:
            continue
        case_id = f"crud-{category}-{row['ID']}"
        if case_id in prior or str(row["ID"]) in seen_events or str(row["ID"]) in source_event_ids:
            continue
        n = int(re.search(r"_(\d)doc", category).group(1))
        docs = {"crud-" + digest(row[f"news{i}"].strip().encode("utf-8"))[:20]: row[f"news{i}"].strip() + "\n" for i in range(1, n + 1)}
        try:
            gold = call_json(GOLD_SYSTEM, {
                "question": row["questions"].strip(), "publisher_reference_answer": row["answers"].strip(),
                "indexed_source_documents": [{"doc_id": doc_id, "text": body} for doc_id, body in docs.items()],
            }, lambda value: validate_gold(value, docs))
            status = "gold_quote_checked_candidate" if gold["gold_supported"] else "publisher_answer_not_supported"
            if status == "gold_quote_checked_candidate" and n > 1:
                # A multi-document case must contribute source-specific evidence
                # from every linked document, not just several copies of a story.
                evidence = gold["evidence"]
                for doc_id, body in docs.items():
                    unique_quotes = [item["quote"] for item in evidence if item["doc_id"] == doc_id
                                     and all(item["quote"] not in other for other_id, other in docs.items() if other_id != doc_id)]
                    if not unique_quotes:
                        status = "redundant_or_missing_source_evidence"
                        break
        except Exception as exc:
            gold = {"error": str(exc)}
            status = "screen_failed"
        item = {"case_id": case_id, "group_id": str(row["ID"]), "category": category, "split": split,
                "status": status, "gold": gold, "source_doc_ids": list(docs),
                "source_sha256": {doc_id: hashlib.sha256(body.encode()).hexdigest() for doc_id, body in docs.items()}}
        audit.append(item)
        write_jsonl(audit_path, audit)
        prior.add(case_id)
        if status == "gold_quote_checked_candidate":
            selected.append(item)
            counts[(category, split)] += 1
            seen_events.add(str(row["ID"]))
            write_jsonl(selected_path, selected)
            print(f"Accepted {case_id}; {dict(counts)}", flush=True)
        elif len(audit) % 10 == 0:
            print(f"Screened {len(audit)}; accepted {len(selected)}/12", flush=True)
        if all(counts[key] == value for key, value in targets.items()):
            break
    if counts != targets:
        raise ValueError(f"Insufficient evidence-screened candidates: {counts} vs {targets}")
    print(f"Selected {len(selected)} replacements after {len(audit)} candidate screens")


def build() -> None:
    bad, old_reviews = invalid_v1_cases()
    selected = rows(V2.parent / "replacement_selection.jsonl")
    if len(selected) != 12 or any(item["status"] != "gold_quote_checked_candidate" for item in selected):
        raise ValueError("Need 12 quote-checked replacements")
    old_by_group = Counter((case["category"], case["split"]) for case in bad if case["case_id"] != REPAIR_ID)
    new_by_group = Counter((item["category"], item["split"]) for item in selected)
    if old_by_group != new_by_group:
        raise ValueError(f"Replacement balance changed: {new_by_group} != {old_by_group}")
    raw = read_json(RAW)
    source = {f"crud-{category}-{row['ID']}": row for n in (1, 2, 3) for category in [f"questanswer_{n}doc" if n == 1 else f"questanswer_{n}docs"] for row in raw[category]}
    original = rows(V1 / "cases.jsonl")
    docs = {item["doc_id"]: item for item in rows(V1 / "documents.jsonl")}
    V2.mkdir(parents=True, exist_ok=True)
    docs_dir = V2 / "documents"
    docs_dir.mkdir(exist_ok=True)
    for item in docs.values():
        shutil.copyfile(V1 / "documents" / item["filename"], docs_dir / item["filename"])
    event_meta = {str(item["ID"]): item for item in raw.get("event_summary", [])}
    new_cases = {}
    for chosen in selected:
        row = source[chosen["case_id"]]
        n = int(re.search(r"_(\d)doc", chosen["category"]).group(1))
        source_ids, evidence = [], []
        for i in range(1, n + 1):
            body = row[f"news{i}"].strip()
            doc_id = "crud-" + digest(body.encode("utf-8"))[:20]
            source_ids.append(doc_id)
            evidence.append({"doc_id": doc_id, **source_evidence_candidate(row["questions"], row["answers"], body)})
            if doc_id not in docs:
                info = event_meta.get(str(row["ID"]), {})
                filename = f"{doc_id}.txt"
                docs[doc_id] = {"doc_id": doc_id, "filename": filename, "role": "source", "source_record_id": str(row["ID"]),
                                "source_url": info.get("url"), "source_title": info.get("title"), "source_time": info.get("time")}
                (docs_dir / filename).write_text(body + "\n", encoding="utf-8")
        new_cases[chosen["case_id"]] = {"case_id": chosen["case_id"], "group_id": str(row["ID"]), "category": chosen["category"],
            "split": chosen["split"], "question": row["questions"].strip(), "reference_answer": row["answers"].strip(),
            "expected_action": "answer", "gold_doc_ids": source_ids, "evidence_candidates": evidence,
            "evidence_review_status": "model_gold_quote_checked_needs_independent_review", "selection_gold_review": chosen["gold"]}
    replacement_map = {}
    for key in sorted(old_by_group):
        old_items = [case for case in bad if case["case_id"] != REPAIR_ID and (case["category"], case["split"]) == key]
        chosen = [item for item in selected if (item["category"], item["split"]) == key]
        for old_case, new_item in zip(old_items, chosen):
            replacement_map[old_case["case_id"]] = new_item["case_id"]
    result = []
    for case in original:
        if case["case_id"] in replacement_map:
            result.append(new_cases[replacement_map[case["case_id"]]])
        elif case["case_id"] == REPAIR_ID:
            repaired = dict(case)
            repaired["publisher_reference_answer"] = repaired["reference_answer"]
            repaired["reference_answer"] = REPAIRED_ANSWER
            repaired["evidence_review_status"] = "corpus_verified_answer_repair_needs_independent_review"
            result.append(repaired)
        else:
            result.append(case)
    if len(result) != 200 or len({item["case_id"] for item in result}) != 200 or len(docs) <= 520:
        raise ValueError("Invalid v2 corpus size or case IDs")
    if Counter(item["category"] for item in result) != Counter(item["category"] for item in original):
        raise ValueError("Question category counts changed")
    if Counter(item["split"] for item in result) != Counter(item["split"] for item in original):
        raise ValueError("Dev/test counts changed")
    texts = {doc_id: normalized((docs_dir / item["filename"]).read_text(encoding="utf-8")) for doc_id, item in docs.items()}
    negatives = [item for item in result if item["category"] == "near_no_answer"]
    literal_conflicts = [{"case_id": item["case_id"], "matching_doc_ids": [doc_id for doc_id, body in texts.items() if normalized(item["reference_answer"]) in body]}
                         for item in negatives]
    literal_conflicts = [item for item in literal_conflicts if item["matching_doc_ids"]]
    write_jsonl(V2 / "cases.jsonl", result)
    write_jsonl(V2 / "documents.jsonl", list(docs.values()))
    write_json(V2.parent / "replacement_map.json", {"replaced": replacement_map, "repaired_answer": {REPAIR_ID: REPAIRED_ANSWER},
        "v1_failed_gold": {case["case_id"]: old_reviews[case["case_id"]]["gold"].get("notes") for case in bad},
        "selection_rule": "same category and split; deterministic candidate shuffle seed 20260926+n; first model-supported original-source quote; no RAG answer used"})
    manifest = {"version": "v2", "upstream_revision": read_json(V1 / "manifest.json")["revision"],
        "question_count": 200, "document_count": len(docs), "category_counts": dict(Counter(item["category"] for item in result)),
        "split_counts": dict(Counter(item["split"] for item in result)), "replacement_count": len(replacement_map), "repaired_answer_count": 1,
        "selection_audit_sha256": hashlib.sha256((V2.parent / "candidate_audit.jsonl").read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256((V2 / "cases.jsonl").read_bytes()).hexdigest(),
        "documents_sha256": hashlib.sha256((V2 / "documents.jsonl").read_bytes()).hexdigest(),
        "negative_literal_conflicts": literal_conflicts,
        "review_boundary": "Model quote validation and Codex audit are not human gold; no-answer requires corpus-boundary review"}
    write_json(V2 / "manifest.json", manifest)
    print(f"Frozen v2: 200 cases, {len(docs)} documents; literal negative conflicts {len(literal_conflicts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("screen", "build"))
    args = parser.parse_args()
    {"screen": screen, "build": build}[args.command]()
