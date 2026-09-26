"""Upload desktop CRUD TXT files via the real API and ask 200 questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

from evaluate import rows, write_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "benchmarks" / "upload200" / "prepared"
RUN = ROOT / "data" / "benchmarks" / "upload200" / "api_run"
DESKTOP = Path("D:/桌面/CRUD-RAG-200题评测/01_待上传文档")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("upload", "answer"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--prepared-root", type=Path, default=PREPARED)
    parser.add_argument("--run-root", type=Path, default=RUN)
    parser.add_argument("--documents-dir", type=Path, default=DESKTOP)
    parser.add_argument("--case-ids-file", type=Path, help="Answer only these case IDs, one per line")
    parser.add_argument("--answers-output", type=Path, help="Override answer JSONL path for a selected run")
    args = parser.parse_args()
    session = requests.Session()
    health = session.get(args.base_url + "/api/health", timeout=15).json()
    if not health.get("model_ready"):
        raise RuntimeError("Project embedding model is not ready")
    prepared = args.prepared_root
    run = args.run_root
    upload_dir = args.documents_dir
    docs = rows(prepared / "documents.jsonl")
    cases = rows(prepared / "cases.jsonl")
    names = {doc["filename"]: doc["doc_id"] for doc in docs}
    run.mkdir(parents=True, exist_ok=True)
    if args.phase == "upload":
        current = session.get(args.base_url + "/api/documents", timeout=20).json()
        existing = {doc["doc_name"] for doc in current["documents"]}
        unexpected = existing - set(names)
        if unexpected:
            raise RuntimeError(f"Isolated store has unexpected documents: {sorted(unexpected)[:5]}")
        log_path = run / "upload_results.jsonl"
        log = {row["doc_name"]: row for row in rows(log_path)} if log_path.exists() else {}
        for position, doc in enumerate(docs, 1):
            filename = doc["filename"]
            if filename in existing:
                continue
            path = upload_dir / filename
            if not path.is_file() or path.read_bytes() != (prepared / "documents" / filename).read_bytes():
                raise ValueError(f"Desktop upload file missing or modified: {filename}")
            with path.open("rb") as source:
                response = session.post(args.base_url + "/api/documents/upload", files={"file": (filename, source, "text/plain")}, timeout=180)
            if response.status_code != 200:
                raise RuntimeError(f"Upload {position}/{len(docs)} {filename}: {response.status_code} {response.text[:400]}")
            item = response.json()
            if item["doc_name"] != filename:
                raise ValueError("Upload API returned a different filename")
            log[filename] = item
            existing.add(filename)
            write_jsonl(log_path, [log[name] for name in names if name in log])
            if position % 25 == 0 or position == len(docs):
                print(f"Uploaded {position}/{len(docs)} documents", flush=True)
        current = session.get(args.base_url + "/api/documents", timeout=20).json()
        if current["document_count"] != len(docs) or {x["doc_name"] for x in current["documents"]} != set(names):
            raise ValueError("Isolated API store does not exactly match the upload corpus")
        write_json(run / "upload_verification.json", {"document_count": current["document_count"], "chunk_count": current["chunk_count"], "source": str(upload_dir), "method": "POST /api/documents/upload"})
        print(f"Verified {current['document_count']} documents, {current['chunk_count']} chunks through project upload API")
        return
    current = session.get(args.base_url + "/api/documents", timeout=20).json()
    if current["document_count"] != len(docs):
        raise ValueError(f"Expected {len(docs)} uploaded docs, found {current['document_count']}")
    if args.case_ids_file:
        wanted = [line.strip() for line in args.case_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(wanted) != len(set(wanted)):
            raise ValueError("Duplicate selected case IDs")
        case_by_id = {case["case_id"]: case for case in cases}
        if set(wanted) - set(case_by_id):
            raise ValueError("Selected IDs are not present in the fixed case set")
        cases = [case_by_id[case_id] for case_id in wanted]
    output = args.answers_output or run / "crud200_api_answers.jsonl"
    previous = rows(output) if output.exists() else []
    if [x["case_id"] for x in previous] != [x["case_id"] for x in cases[:len(previous)]]:
        raise ValueError("Answer resume log is not a prefix of fixed cases")
    results = list(previous)
    for position, case in enumerate(cases, 1):
        if position <= len(previous):
            continue
        response = session.post(args.base_url + "/api/chat", json={"question": case["question"]}, timeout=180)
        if response.status_code != 200:
            raise RuntimeError(f"Question {position}/{len(cases)} {case['case_id']}: {response.status_code} {response.text[:400]}")
        item = response.json()
        final = [names[source["doc_name"]] for source in item["sources"]]
        debug = item.get("debug") or {}
        coarse = [names[source["doc_name"]] for source in debug.get("candidates", [])]
        gold = case["gold_doc_ids"]
        results.append({
            "case_id": case["case_id"], "track": "crud_upload_api", "split": case["split"],
            "category": case["category"], "expected_action": case["expected_action"],
            "label_status": case.get("evidence_review_status"), "question": case["question"],
            "reference_answer": case["reference_answer"], "gold_doc_ids": gold,
            "evidence_candidates": case["evidence_candidates"], "coarse_doc_ids": coarse,
            "final_doc_ids": final,
            "coarse_recall_at_20": len(set(coarse[:20]) & set(gold)) / len(set(gold)) if gold else None,
            "final_recall_at_8": len(set(final[:8]) & set(gold)) / len(set(gold)) if gold else None,
            "answer": item["answer"], "sources": item["sources"],
            "retrieval_count": item["retrieval_count"], "project_timestamp": item.get("timestamp"),
        })
        write_jsonl(output, results)
        if position % 10 == 0 or position == len(cases):
            print(f"Answered {position}/{len(cases)} cases", flush=True)
    verification = output.with_suffix(".verification.json") if args.case_ids_file else run / "answer_verification.json"
    write_json(verification, {"cases": len(results), "uploaded_documents": current["document_count"], "method": "POST /api/chat", "distinct_case_ids": len({x["case_id"] for x in results}), "selected_run": bool(args.case_ids_file)})
    print(f"Verified {len(results)} answers through project chat API")


if __name__ == "__main__":
    main()
