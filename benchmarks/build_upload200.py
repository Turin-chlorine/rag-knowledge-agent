"""Extend the fixed CRUD set to 200 cases for the real upload API run.

Keep the original 140 cases and 400 source/background TXT files intact. Add
20 publisher QA cases from each answerable category and the news texts they
cite. Questions and reference answers are saved outside the upload directory.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from prepare import bigrams, crud_rows, digest, normalized, read_json, source_evidence_candidate, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "benchmarks"
OLD = BASE / "crud" / "prepared"
RAW = BASE / "crud" / "raw" / "split_merged.json"
OUT = BASE / "upload200" / "prepared"


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    raw = read_json(RAW)
    old_cases = lines(OLD / "cases.jsonl")
    old_docs = lines(OLD / "documents.jsonl")
    if len(old_cases) != 140 or len(old_docs) != 400:
        raise ValueError("Expected the pinned CRUD 140-case, 400-document baseline")
    old_events = {case["group_id"] for case in old_cases}
    old_ids = {case["case_id"] for case in old_cases}
    selected = []
    counts = Counter()
    for category, row in crud_rows(raw, per_type=80, seed=20260926):
        if counts[category] >= 20 or str(row["ID"]) in old_events or f"crud-{category}-{row['ID']}" in old_ids:
            continue
        selected.append((category, row))
        counts[category] += 1
    if sorted(counts.values()) != [20, 20, 20]:
        raise ValueError(f"Could not get 20 disjoint cases per category: {counts}")
    docs_dir = OUT / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    documents = {doc["doc_id"]: doc for doc in old_docs}
    for doc in old_docs:
        source = OLD / "documents" / doc["filename"]
        target = docs_dir / doc["filename"]
        if source.read_bytes() != target.read_bytes() if target.exists() else True:
            shutil.copyfile(source, target)
    metadata = {str(row["ID"]): row for row in raw.get("event_summary", [])}
    new_cases = []
    for category, row in selected:
        source_count = int(re.search(r"_(\d)doc", category).group(1))
        doc_ids, evidence = [], []
        for index in range(1, source_count + 1):
            body = row[f"news{index}"].strip()
            doc_id = "crud-" + digest(body.encode("utf-8"))[:20]
            doc_ids.append(doc_id)
            evidence.append({"doc_id": doc_id, **source_evidence_candidate(row["questions"], row["answers"], body)})
            if doc_id not in documents:
                info = metadata.get(str(row["ID"]), {})
                filename = f"{doc_id}.txt"
                documents[doc_id] = {
                    "doc_id": doc_id, "filename": filename, "role": "source",
                    "source_record_id": str(row["ID"]), "source_url": info.get("url"),
                    "source_title": info.get("title"), "source_time": info.get("time"),
                }
                (docs_dir / filename).write_text(body + "\n", encoding="utf-8")
        case_id = f"crud-{category}-{row['ID']}"
        new_cases.append({
            "case_id": case_id, "group_id": str(row["ID"]), "category": category,
            "question": row["questions"].strip(), "reference_answer": row["answers"].strip(),
            "expected_action": "answer", "gold_doc_ids": doc_ids,
            "evidence_candidates": evidence, "evidence_review_status": "needs_human_review",
            "split": "test" if int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 4 == 0 else "dev",
        })
    # Adding source files can invalidate a former near-no-answer label. Preserve
    # the question, but explicitly flag any newly present literal answer.
    texts = {doc_id: normalized((docs_dir / doc["filename"]).read_text(encoding="utf-8")) for doc_id, doc in documents.items()}
    negatives_with_literal_answer = []
    for case in old_cases:
        if case["category"] == "near_no_answer":
            matches = [doc_id for doc_id, body in texts.items() if normalized(case["reference_answer"]) in body]
            if matches:
                negatives_with_literal_answer.append({"case_id": case["case_id"], "matching_doc_ids": matches})
                case["evidence_review_status"] = "label_invalid_literal_answer_in_expanded_corpus"
    cases = old_cases + new_cases
    if len(cases) != 200 or len({case["case_id"] for case in cases}) != 200:
        raise ValueError("Expected 200 unique CRUD cases")
    write_jsonl(OUT / "documents.jsonl", list(documents.values()))
    write_jsonl(OUT / "cases.jsonl", cases)
    write_json(OUT / "manifest.json", {
        "source": "IAAR-Shanghai/CRUD_RAG", "revision": "1aace383994e1f68efa12cf2a8e2dadfb4102ceb",
        "base_case_count": 140, "added_case_count": 60, "question_count": len(cases),
        "document_count": len(documents), "categories": dict(Counter(case["category"] for case in cases)),
        "splits": dict(Counter(case["split"] for case in cases)),
        "questions_and_answers_in_uploaded_text": False,
        "old_negatives_with_literal_answer_after_expansion": negatives_with_literal_answer,
        "source_file_sha256": digest(RAW.read_bytes()),
        "document_list_sha256": digest((OUT / "documents.jsonl").read_bytes()),
        "case_list_sha256": digest((OUT / "cases.jsonl").read_bytes()),
        "label_status": "publisher source links; semantic gold and no-answer boundary require independent review",
    })
    print(f"Prepared {len(cases)} cases, {len(documents)} TXT files; literal negative-label conflicts: {len(negatives_with_literal_answer)}")


if __name__ == "__main__":
    main()
