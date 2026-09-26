"""Prepare small, source-pinned Chinese RAG benchmark tracks.

All downloaded content and generated cases live under data/benchmarks/ (gitignored).
No benchmark question or reference answer is written into an indexed document.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "benchmarks"
CRUD_REV = "1aace383994e1f68efa12cf2a8e2dadfb4102ceb"
RGB_REV = "65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615"
MIRACL_CORPUS_REV = "a7362f0"  # Hugging Face dataset revision shown by publisher
MIRACL_QRELS_REV = "5be20db9509754dadad47689368639fcec739c00"
CRUD_BASE = f"https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/{CRUD_REV}/data"
RGB_BASE = f"https://raw.githubusercontent.com/chen700564/RGB/{RGB_REV}/data"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "rag-knowledge-agent-evaluation"})
        with urllib.request.urlopen(request, timeout=90) as response, path.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
    return path


def read_json(path: Path):
    content = path.read_text(encoding="utf-8")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def bigrams(text: str) -> set[str]:
    text = normalized(text)
    return {text[i:i + 2] for i in range(len(text) - 1)}


def source_evidence_candidate(question: str, answer: str, text: str) -> dict:
    """Select a verbatim candidate span; this is not a human-confirmed label."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"[^。！？!?\n]+[。！？!?]?", text)]
    if not spans:
        spans = [(0, len(text))]
    terms = set(re.findall(r"[\u4e00-\u9fff]{2}|[A-Za-z0-9]{2,}", question + answer))
    best = max(spans, key=lambda pair: (len(terms & set(re.findall(r"[\u4e00-\u9fff]{2}|[A-Za-z0-9]{2,}", text[pair[0]:pair[1]]))), -pair[0]))
    start, end = best
    return {"start": start, "end": end, "quote": text[start:end].strip(), "review_status": "needs_human_review"}


def crud_rows(raw: dict, *, per_type: int, seed: int) -> list[tuple[str, dict]]:
    rng = random.Random(seed)
    selected = []
    used_events = set()
    used_questions = set()
    for doc_count in (1, 2, 3):
        category = f"questanswer_{doc_count}doc" if doc_count == 1 else f"questanswer_{doc_count}docs"
        rows = list(raw[category])
        rng.shuffle(rows)
        added = 0
        for row in rows:
            event = str(row["ID"])
            question = normalized(row["questions"])
            if event in used_events or question in used_questions or not row.get("answers", "").strip():
                continue
            if any(not row.get(f"news{i}", "").strip() for i in range(1, doc_count + 1)):
                continue
            selected.append((category, row))
            used_events.add(event)
            used_questions.add(question)
            added += 1
            if added == per_type:
                break
        if added != per_type:
            raise ValueError(f"Only {added} distinct {category} rows; requested {per_type}")
    return selected


def prepare_crud(args: argparse.Namespace) -> None:
    root = args.root / "crud"
    raw_dir = root / "raw"
    split_url = f"{CRUD_BASE}/crud_split/split_merged.json"
    shard_url = f"{CRUD_BASE}/80000_docs/documents_dup_part_1_part_1"
    split_path = download(split_url, raw_dir / "split_merged.json")
    shard_path = download(shard_url, raw_dir / "documents_dup_part_1_part_1")
    raw = read_json(split_path)
    selected = crud_rows(raw, per_type=args.questions_per_type, seed=args.seed)
    documents: dict[str, dict] = {}
    cases: list[dict] = []
    metadata = {str(row["ID"]): row for row in raw.get("event_summary", [])}
    for category, row in selected:
        required = int(re.search(r"_(\d)doc", category).group(1))
        ids = []
        candidates = []
        for index in range(1, required + 1):
            body = row[f"news{index}"].strip()
            doc_id = "crud-" + digest(body.encode("utf-8"))[:20]
            ids.append(doc_id)
            evidence = source_evidence_candidate(row["questions"], row["answers"], body)
            candidates.append({"doc_id": doc_id, **evidence})
            if doc_id not in documents:
                info = metadata.get(str(row["ID"]), {})
                documents[doc_id] = {
                    "doc_id": doc_id,
                    "filename": f"{doc_id}.txt",
                    "text": body,
                    "role": "source",
                    "source_record_id": str(row["ID"]),
                    "source_url": info.get("url"),
                    "source_title": info.get("title"),
                    "source_time": info.get("time"),
                }
        cases.append({
            "case_id": f"crud-{category}-{row['ID']}",
            "group_id": str(row["ID"]),
            "category": category,
            "question": row["questions"].strip(),
            "reference_answer": row["answers"].strip(),
            "expected_action": "answer",
            "gold_doc_ids": ids,
            "evidence_candidates": candidates,
            "evidence_review_status": "needs_human_review",
        })
    if len(documents) > args.doc_count:
        raise ValueError(f"{len(documents)} source documents exceed --doc-count={args.doc_count}")

    # A fixed random sample from the publisher's separate retrieval corpus adds
    # background noise. The question-bearing source texts stay untouched.
    rng = random.Random(args.seed)
    distractors = [line.strip() for line in shard_path.read_text(encoding="utf-8").splitlines() if len(line.strip()) >= 80]
    question_grams = [bigrams(case["question"]) for case in cases]
    background_needed = args.doc_count - len(documents)
    ranked_distractors = sorted(distractors, key=lambda body: -max((len(bigrams(body[:160]) & gram) / max(1, len(gram)) for gram in question_grams), default=0))
    hard_distractors = ranked_distractors[:background_needed // 2]
    hard_texts = set(hard_distractors)
    rng.shuffle(distractors)
    chosen = hard_distractors + [body for body in distractors if body not in hard_texts]
    hard_hashes = {digest(body.encode("utf-8"))[:20] for body in hard_distractors}
    for body in chosen:
        if len(documents) >= args.doc_count:
            break
        doc_id = "crud-" + digest(body.encode("utf-8"))[:20]
        if doc_id not in documents:
            role = "background_hard" if doc_id.removeprefix("crud-") in hard_hashes else "background_random"
            documents[doc_id] = {"doc_id": doc_id, "filename": f"{doc_id}.txt", "text": body, "role": role, "source_record_id": None, "source_url": None, "source_title": None, "source_time": None}
    if len(documents) != args.doc_count:
        raise ValueError("Not enough distinct source and background documents")

    # Candidate near-boundary questions come from other publisher QA records.
    # They are not gold negatives until a human confirms that no selected
    # document supports their answer. Never count them in refusal metrics first.
    selected_events = {case["group_id"] for case in cases}
    selected_questions = [bigrams(case["question"]) for case in cases]
    negative_pool = []
    for row in raw["questanswer_1doc"]:
        if str(row["ID"]) in selected_events:
            continue
        query = row.get("questions", "").strip()
        if not query or not row.get("answers", "").strip():
            continue
        gram = bigrams(query)
        proximity = max((len(gram & other) / max(1, len(gram | other)) for other in selected_questions), default=0)
        negative_pool.append((proximity, str(row["ID"]), row))
    negative_pool.sort(key=lambda item: (-item[0], item[1]))
    normalized_documents = [(doc_id, normalized(doc["text"])) for doc_id, doc in documents.items()]
    negative_added = 0
    for _, event, row in negative_pool:
        if any(normalized(row["answers"]) in body for _, body in normalized_documents):
            continue
        cases.append({
            "case_id": f"crud-near-no-answer-{event}", "group_id": event,
            "category": "near_no_answer", "question": row["questions"].strip(),
            "reference_answer": row["answers"].strip(), "expected_action": "refuse",
            "gold_doc_ids": [], "evidence_candidates": [],
            "evidence_review_status": "needs_human_review",
            "note": "Candidate only: verify the answer is absent from every indexed document before scoring.",
        })
        negative_added += 1
        if negative_added == args.negative_questions:
            break
    if negative_added != args.negative_questions:
        raise ValueError(f"Only {negative_added} near-no-answer candidates passed the exact-answer exclusion")

    # Hold out whole source events, not individual questions. Cases remain
    # source-linked; no-answer examples require explicit human verification.
    for ordinal, category in enumerate(sorted({case["category"] for case in cases})):
        same_type = [case for case in cases if case["category"] == category]
        group_ids = sorted({case["group_id"] for case in same_type})
        random.Random(args.seed + ordinal).shuffle(group_ids)
        test_groups = set(group_ids[: max(1, round(len(group_ids) * args.test_fraction))])
        for case in same_type:
            case["split"] = "test" if case["group_id"] in test_groups else "dev"
    doc_splits: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        for doc_id in case["gold_doc_ids"]:
            doc_splits[doc_id].add(case["split"])
    if any(len(splits) > 1 for splits in doc_splits.values()):
        raise ValueError("A source document appears in both dev and test cases")
    output = root / "prepared"
    docs_dir = output / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    expected_files = {doc["filename"] for doc in documents.values()}
    for old_file in docs_dir.glob("*.txt"):
        if old_file.name not in expected_files:
            old_file.unlink()
    for doc in documents.values():
        (docs_dir / doc["filename"]).write_text(doc["text"] + "\n", encoding="utf-8")
    write_jsonl(output / "documents.jsonl", [{k: v for k, v in doc.items() if k != "text"} for doc in documents.values()])
    write_jsonl(output / "cases.jsonl", cases)
    negative_audit = []
    for case in cases:
        if case["category"] != "near_no_answer":
            continue
        query_grams = bigrams(case["question"])
        ranked_docs = sorted(documents.values(), key=lambda doc: -len(query_grams & bigrams(doc["text"][:300])))[:15]
        negative_audit.append({
            "case_id": case["case_id"],
            "exact_reference_answer_matches": [doc_id for doc_id, body in normalized_documents if normalized(case["reference_answer"]) in body],
            "most_similar_documents_to_review": [{"doc_id": doc["doc_id"], "filename": doc["filename"], "role": doc["role"]} for doc in ranked_docs],
            "status": "needs_human_review",
        })
    write_jsonl(output / "near_no_answer_audit.jsonl", negative_audit)
    write_jsonl(output / "gold_review_template.jsonl", [{
        "case_id": case["case_id"], "category": case["category"],
        "question": case["question"], "reference_answer": case["reference_answer"],
        "gold_doc_ids": case["gold_doc_ids"],
        "evidence_candidates": case["evidence_candidates"],
        "gold_evidence_verified": None, "verified_evidence": [],
        "no_answer_verified": None, "reviewer": "", "review_notes": "",
    } for case in cases])
    write_json(output / "manifest.json", {
        "track": "crud", "source": "IAAR-Shanghai/CRUD_RAG", "revision": CRUD_REV,
        "source_files": [{"url": url, "sha256": digest(path.read_bytes())} for url, path in ((split_url, split_path), (shard_url, shard_path))],
        "seed": args.seed, "document_count": len(documents), "question_count": len(cases),
        "background_selection": "half title/lead lexical-near, half deterministic random from one upstream shard",
        "source_provenance": {"source_documents": sum(doc["role"] == "source" for doc in documents.values()), "source_documents_with_upstream_url": sum(doc["role"] == "source" and bool(doc["source_url"]) for doc in documents.values()), "all_documents_have_upstream_record_or_shard": True},
        "question_counts": {category: sum(c["category"] == category for c in cases) for category in sorted({c["category"] for c in cases})},
        "splits": {split: sum(c["split"] == split for c in cases) for split in ("dev", "test")},
        "annotation_status": "source_docs_linked; passage evidence and answer claims require human review",
        "license_status": "CRUD-RAG repository has no declared license; check upstream news-text rights before redistribution",
        "comparable_to_full_benchmark": False,
    })
    print(f"CRUD: {len(documents)} documents, {len(cases)} cases -> {output}")


def read_lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        yield from source


def prepare_miracl(args: argparse.Namespace) -> None:
    root = args.root / "miracl"
    raw_dir = root / "raw"
    hf = f"https://huggingface.co/datasets/miracl/miracl/resolve/{MIRACL_QRELS_REV}/miracl-v1.0-zh"
    topics_url = f"{hf}/topics/topics.miracl-v1.0-zh-dev.tsv"
    qrels_url = f"{hf}/qrels/qrels.miracl-v1.0-zh-dev.tsv"
    topics_path = download(topics_url, raw_dir / "topics.tsv")
    qrels_path = download(qrels_url, raw_dir / "qrels.tsv")
    corpus_url = f"https://huggingface.co/datasets/miracl/miracl-corpus/resolve/{MIRACL_CORPUS_REV}/miracl-corpus-v1.0-zh/docs-0.jsonl.gz"
    corpus_path = args.corpus or download(corpus_url, raw_dir / "docs-0.jsonl.gz")
    topics = dict(line.rstrip("\n").split("\t", 1) for line in read_lines(topics_path) if "\t" in line)
    judgments: dict[str, dict[str, int]] = defaultdict(dict)
    for line in read_lines(qrels_path):
        qid, _, docid, relevance = line.split()
        judgments[qid][docid] = int(relevance)
    corpus = {}
    for line in read_lines(corpus_path):
        row = json.loads(line)
        # The app's splitter discards fragments below 20 characters. Exclude
        # such passages before selecting judged items, rather than silently
        # losing a positive document during indexing.
        if len((row.get("title", "") + row.get("text", "")).strip()) >= 40:
            corpus[row["docid"]] = row
    eligible = [qid for qid in topics if any(rel > 0 and docid in corpus for docid, rel in judgments[qid].items())]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    selected = eligible[:args.questions]
    if len(selected) < args.questions:
        raise ValueError(f"Selected corpus shard contains positive passages for only {len(selected)} queries; requested {args.questions}")
    needed = {docid for qid in selected for docid in judgments[qid] if docid in corpus}
    if len(needed) > args.doc_count:
        raise ValueError(f"{len(needed)} judged passages exceed --doc-count={args.doc_count}")
    others = [docid for docid in corpus if docid not in needed]
    gold_articles = {docid.split("#", 1)[0] for qid in selected for docid, rel in judgments[qid].items() if rel > 0}
    question_grams = [bigrams(topics[qid]) for qid in selected]
    candidates = rng.sample(others, min(20000, len(others)))
    ranked = []
    for docid in candidates:
        if docid.split("#", 1)[0] in gold_articles:
            continue
        title_grams = bigrams(corpus[docid]["title"])
        score = max((len(title_grams & gram) / max(1, len(gram)) for gram in question_grams), default=0)
        ranked.append((score, docid))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    remainder_count = args.doc_count - len(needed)
    hard = [docid for score, docid in ranked if score > 0][:remainder_count // 2]
    hard_set = set(hard)
    rng.shuffle(others)
    random_background = [docid for docid in others if docid not in hard_set][:remainder_count - len(hard)]
    ids = sorted(needed) + hard + random_background
    if len(ids) != args.doc_count:
        raise ValueError("Not enough corpus passages")
    output = root / "prepared"
    docs_dir = output / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    expected_files = {"miracl-" + digest(docid.encode("utf-8"))[:20] + ".txt" for docid in ids}
    for old_file in docs_dir.glob("*.txt"):
        if old_file.name not in expected_files:
            old_file.unlink()
    docs = []
    for docid in ids:
        row = corpus[docid]
        filename = "miracl-" + digest(docid.encode("utf-8"))[:20] + ".txt"
        (docs_dir / filename).write_text(row["title"] + "\n" + row["text"] + "\n", encoding="utf-8")
        role = "judged" if docid in needed else "unjudged_hard" if docid in hard_set else "unjudged_random"
        docs.append({"doc_id": docid, "filename": filename, "role": role, "source_title": row["title"]})
    cases = []
    for qid in selected:
        scoped = {docid: rel for docid, rel in judgments[qid].items() if docid in corpus}
        cases.append({"case_id": "miracl-" + qid, "group_id": qid, "category": "retrieval", "question": topics[qid], "gold_doc_ids": [docid for docid, rel in scoped.items() if rel > 0], "judged_negative_doc_ids": [docid for docid, rel in scoped.items() if rel <= 0], "expected_action": "retrieve", "split": "test"})
    write_jsonl(output / "documents.jsonl", docs)
    write_jsonl(output / "cases.jsonl", cases)
    write_json(output / "manifest.json", {"track": "miracl", "source": "project-miracl/miracl", "upstream_split": "dev", "local_split": "test", "qrels_revision": MIRACL_QRELS_REV, "corpus_revision": MIRACL_CORPUS_REV, "seed": args.seed, "document_count": len(docs), "question_count": len(cases), "hard_unjudged_documents": len(hard), "source_files": [{"url": url, "sha256": digest(path.read_bytes())} for url, path in ((topics_url, topics_path), (qrels_url, qrels_path), (corpus_url if not args.corpus else str(args.corpus), corpus_path))], "annotation_status": "publisher qrels; queries filtered to positives in selected corpus shard; half of extra passages are title-near unjudged distractors, not assumed negative", "comparable_to_full_benchmark": False})
    print(f"MIRACL: {len(docs)} passages, {len(cases)} queries -> {output}")


def prepare_rgb(args: argparse.Namespace) -> None:
    root = args.root / "rgb"
    raw_dir = root / "raw"
    files = {}
    for name in ("zh_refine.json", "zh_int.json", "zh_fact.json"):
        url = f"{RGB_BASE}/{name}"
        path = download(url, raw_dir / name)
        files[name] = (url, path)
    rng = random.Random(args.seed)
    cases = []
    for name, kind in (("zh_refine.json", "noise_rejection"), ("zh_int.json", "integration"), ("zh_fact.json", "counterfactual")):
        rows = read_json(files[name][1])
        rng.shuffle(rows)
        for row in rows[:args.questions_per_type]:
            positive = row.get("positive", [])
            negative = row.get("negative", [])
            if kind == "counterfactual":
                contexts = positive[:2] + row.get("positive_wrong", [])[:2] + negative[:1]
                expected = "identify_conflict_or_refuse"
            elif kind == "noise_rejection":
                contexts = negative[:5]
                expected = "refuse"
            else:
                contexts = positive[:3] + negative[:2]
                expected = "answer"
            if not contexts:
                continue
            cases.append({"case_id": f"rgb-{kind}-{row['id']}", "group_id": str(row["id"]), "category": kind, "question": row["query"], "reference_answer": row.get("answer"), "expected_action": expected, "contexts": contexts, "split": "test", "review_status": "needs_human_review"})
    output = root / "prepared"
    write_jsonl(output / "cases.jsonl", cases)
    write_json(output / "manifest.json", {"track": "rgb", "source": "chen700564/RGB", "revision": RGB_REV, "seed": args.seed, "case_count": len(cases), "source_files": [{"url": url, "sha256": digest(path.read_bytes())} for url, path in files.values()], "license": "CC BY-NC-SA 4.0; noncommercial use only", "mode": "provided-context generation stress test, not corpus-wide retrieval", "comparable_to_full_benchmark": False})
    print(f"RGB: {len(cases)} context tests -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", choices=("crud", "miracl", "rgb"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--doc-count", type=int, default=400)
    parser.add_argument("--questions-per-type", type=int, default=40)
    parser.add_argument("--questions", type=int, default=80)
    parser.add_argument("--negative-questions", type=int, default=20)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--corpus", type=Path, help="Existing MIRACL zh corpus JSONL or JSONL.GZ; avoids download")
    args = parser.parse_args()
    if args.track == "crud":
        prepare_crud(args)
    elif args.track == "miracl":
        prepare_miracl(args)
    else:
        prepare_rgb(args)


if __name__ == "__main__":
    main()
