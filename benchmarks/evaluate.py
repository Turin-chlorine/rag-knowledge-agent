"""Build isolated benchmark indexes, run the real RAG pipeline, and score reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "benchmarks"
BACKEND = ROOT / "backend"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(value, ensure_ascii=False) + "\n")


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_fingerprint(prepared: Path, docs: list[dict]) -> str:
    combined = hashlib.sha256()
    for doc in docs:
        combined.update(doc["doc_id"].encode("utf-8"))
        combined.update(bytes.fromhex(fingerprint(prepared / "documents" / doc["filename"])))
    return combined.hexdigest()


def backend_imports():
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    import config
    import embedding
    import text_splitter
    import vector_store
    return config, embedding, text_splitter, vector_store


def track_dir(root: Path, track: str) -> Path:
    return root / track / "prepared"


def index_track(root: Path, track: str, force: bool = False, max_docs: int | None = None) -> Path:
    if track == "rgb":
        raise ValueError("RGB tests supplied contexts and does not need an index")
    prepared = track_dir(root, track)
    docs = rows(prepared / "documents.jsonl")
    if max_docs is not None:
        docs = docs[:max_docs]
    config, embedding, text_splitter, vector_store = backend_imports()
    index = root / track / "index" / ("smoke-vector-store.json" if max_docs else "vector-store.json")
    stamp = index.with_suffix(".manifest.json")
    expected = {
        "documents_sha256": fingerprint(prepared / "documents.jsonl"),
        "corpus_sha256": corpus_fingerprint(prepared, docs),
        "count": len(docs), "embedding_model": config.EMBEDDING_MODEL,
        "chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP,
    }
    if index.exists() and stamp.exists() and not force:
        existing = json.loads(stamp.read_text(encoding="utf-8"))
        if existing == expected:
            print(f"Index current: {index}")
            return index
        if "corpus_sha256" not in existing and all(existing.get(key) == value for key, value in expected.items() if key != "corpus_sha256"):
            # Upgrade older index metadata only after checking every indexed
            # chunk against the current source files, without re-embedding.
            stored = json.loads(index.read_text(encoding="utf-8"))
            by_doc = defaultdict(list)
            for chunk in stored.get("chunks", []):
                by_doc[chunk["doc_id"]].append(chunk)
            if {doc["doc_id"] for doc in docs} == set(by_doc):
                intact = True
                for doc in docs:
                    text = (prepared / "documents" / doc["filename"]).read_text(encoding="utf-8")
                    pieces = text_splitter.split_text_into_chunks(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                    indexed_pieces = [chunk["content"] for chunk in sorted(by_doc[doc["doc_id"]], key=lambda item: item["chunk_index"])]
                    if pieces != indexed_pieces:
                        intact = False
                        break
                if intact:
                    write_json(stamp, expected)
                    print(f"Index verified and metadata updated: {index}")
                    return index
    index.parent.mkdir(parents=True, exist_ok=True)
    metadata = []
    chunks = []
    for position, doc in enumerate(docs, 1):
        text = (prepared / "documents" / doc["filename"]).read_text(encoding="utf-8")
        pieces = text_splitter.split_text_into_chunks(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not pieces:
            raise ValueError(f"No chunks: {doc['filename']}")
        vectors = embedding.embed_texts(pieces)
        for number, (piece, vector) in enumerate(zip(pieces, vectors)):
            chunks.append(asdict(vector_store.DocumentChunk(
                chunk_id=hashlib.sha256(f"{doc['doc_id']}:{number}:{piece}".encode()).hexdigest()[:20],
                doc_id=doc["doc_id"], doc_name=doc["filename"], content=piece,
                chunk_index=number, vector=vector,
            )))
        metadata.append(asdict(vector_store.DocumentMeta(
            doc_id=doc["doc_id"], doc_name=doc["filename"],
            chunk_count=len(pieces), char_count=len(text), upload_time="benchmark",
        )))
        if position % 25 == 0 or position == len(docs):
            print(f"Indexed {position}/{len(docs)} documents", flush=True)
    temporary = index.with_suffix(".tmp")
    write_json(temporary, {"documents": metadata, "chunks": chunks})
    temporary.replace(index)
    write_json(stamp, expected)
    print(f"Index ready: {index} ({len(chunks)} chunks)")
    return index


def recall(found: list[str], expected: list[str]) -> float | None:
    if not expected:
        return None
    return len(set(found) & set(expected)) / len(set(expected))


def rank_metrics(ranked: list[str], gold: list[str]) -> dict:
    unique = list(dict.fromkeys(ranked))
    out = {"recall_at_20": recall(unique[:20], gold), "recall_at_8": recall(unique[:8], gold)}
    positions = [position for position, docid in enumerate(unique, 1) if docid in gold]
    out["first_relevant_rank"] = positions[0] if positions else None
    return out


def retrieval_run(root: Path, track: str, split: str, limit: int | None, e2e: bool, case_ids_file: Path | None = None, resume_results: Path | None = None) -> Path:
    prepared = track_dir(root, track)
    selected = [case for case in rows(prepared / "cases.jsonl") if case["split"] == split]
    if case_ids_file:
        ids = [line.strip() for line in case_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs in --case-ids-file")
        by_id = {case["case_id"]: case for case in selected}
        missing = set(ids) - set(by_id)
        if missing:
            raise ValueError(f"Case IDs absent from {split}: {sorted(missing)[:5]}")
        selected = [by_id[case_id] for case_id in ids]
    if limit:
        selected = selected[:limit]
    index = root / track / "index" / "vector-store.json"
    if not index.exists():
        raise FileNotFoundError(f"Build the index first: python benchmarks/evaluate.py index {track}")
    config, _, _, vector_store = backend_imports()
    stamp = index.with_suffix(".manifest.json")
    indexed = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
    if indexed.get("corpus_sha256") != corpus_fingerprint(prepared, rows(prepared / "documents.jsonl")) or indexed.get("embedding_model") != config.EMBEDDING_MODEL or indexed.get("chunk_size") != config.CHUNK_SIZE or indexed.get("chunk_overlap") != config.CHUNK_OVERLAP:
        raise ValueError("Index does not match current corpus/model/chunk settings; run index again")
    import agent
    if e2e and not config.is_api_key_configured():
        raise RuntimeError("DeepSeek API key missing; use retrieval mode or configure .env")
    store = vector_store.VectorStore(index)
    agent.vector_store = store  # Use the production retrieval and answer path on an isolated index.
    filename_to_id = {doc["filename"]: doc["doc_id"] for doc in rows(prepared / "documents.jsonl")}
    mode = "e2e" if e2e else "retrieval"
    suffix = f"_selected{len(selected)}" if case_ids_file else ""
    suffix += f"_limit{limit}" if limit else ""
    if e2e:
        suffix += "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = resume_results or root / track / "runs" / f"{split}_{mode}{suffix}.jsonl"
    if resume_results:
        if not e2e or not output.exists() or output.resolve().parent != (root / track / "runs").resolve():
            raise ValueError("--resume-results must name an existing e2e result in this track's runs directory")
    previous = rows(output) if resume_results else []
    prior_ids = [row["case_id"] for row in previous]
    if len(prior_ids) != len(set(prior_ids)) or prior_ids != [case["case_id"] for case in selected[:len(previous)]]:
        raise ValueError("Existing results are not a prefix of the selected cases")
    results = list(previous)
    for number, case in enumerate(selected, 1):
        if number <= len(previous):
            continue
        if e2e:
            response = agent.agent_answer(case["question"])
            final = [filename_to_id[src["doc_name"]] for src in response["sources"]]
            debug = response["debug"]
            answer = response["answer"]
            sources = response["sources"]
        else:
            hits, debug = agent.retrieve_documents(case["question"])
            final = [filename_to_id[hit["doc_name"]] for hit in hits]
            answer = None
            sources = []
        coarse = [filename_to_id[c["doc_name"]] for c in debug["candidates"]]
        gold = case.get("gold_doc_ids", [])
        results.append({
            "case_id": case["case_id"], "track": track, "split": split,
            "category": case["category"], "expected_action": case["expected_action"],
            "label_status": case.get("evidence_review_status", "publisher_qrels"),
            "question": case["question"], "reference_answer": case.get("reference_answer"),
            "gold_doc_ids": gold, "evidence_candidates": case.get("evidence_candidates", []),
            "coarse_doc_ids": coarse, "final_doc_ids": final,
            "coarse_recall_at_20": recall(coarse[:20], gold),
            "final_recall_at_8": recall(final[:8], gold),
            "first_relevant_rank": rank_metrics(final, gold)["first_relevant_rank"],
            "rerank_applied": debug["rerank_applied"],
            "answer": answer, "sources": sources,
        })
        if e2e:
            write_jsonl(output, results)  # Preserve paid responses if a later case fails.
        if number % 10 == 0 or number == len(selected):
            print(f"Ran {number}/{len(selected)} cases", flush=True)
    write_jsonl(output, results)
    write_retrieval_report(output, results)
    write_json(output.with_suffix(".manifest.json"), {
        "track": track, "split": split, "mode": mode, "cases": len(results),
        "limited_run": limit is not None or case_ids_file is not None,
        "case_ids_file_sha256": fingerprint(case_ids_file) if case_ids_file else None,
        "dataset_sha256": fingerprint(prepared / "manifest.json"),
        "documents_sha256": fingerprint(prepared / "documents.jsonl"),
        "cases_sha256": fingerprint(prepared / "cases.jsonl"),
        "index_sha256": fingerprint(index), "embedding_model": config.EMBEDDING_MODEL,
        "rerank_model": config.RERANK_MODEL, "rerank_enabled": config.RERANK_ENABLED,
        "candidate_k": config.RETRIEVE_CANDIDATE_K, "top_k": config.TOP_K,
        "relevance_threshold": config.RELEVANCE_THRESHOLD,
        "model": config.DEEPSEEK_MODEL if e2e else None,
        "temperature": config.DEEPSEEK_TEMPERATURE if e2e else None,
    })
    if e2e:
        write_jsonl(output.with_name(output.stem + "_review_template.jsonl"), [review_template(row) for row in results])
    print(f"Results: {output}")
    return output


def rgb_run(root: Path, limit: int | None) -> Path:
    prepared = track_dir(root, "rgb")
    cases = rows(prepared / "cases.jsonl")
    if limit:
        cases = cases[:limit]
    config, _, _, _ = backend_imports()
    if not config.is_api_key_configured():
        raise RuntimeError("DeepSeek API key missing; RGB generation run requires .env")
    import agent
    suffix = f"_limit{limit}" if limit else ""
    suffix += "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "rgb" / "runs" / f"test_e2e{suffix}.jsonl"
    results = []
    for number, case in enumerate(cases, 1):
        contexts = [{"doc_name": f"rgb-{case['case_id']}-{i}.txt", "chunk_index": 0, "content": value} for i, value in enumerate(case["contexts"])]
        answer = agent.call_deepseek(case["question"], agent.build_context(contexts), [])
        results.append({"case_id": case["case_id"], "track": "rgb", "split": "test", "category": case["category"], "question": case["question"], "reference_answer": case.get("reference_answer"), "expected_action": case["expected_action"], "contexts": case["contexts"], "answer": answer, "label_status": "needs_human_review"})
        write_jsonl(output, results)
        print(f"Ran RGB {number}/{len(cases)}", flush=True)
    write_jsonl(output, results)
    write_json(output.with_suffix(".manifest.json"), {
        "track": "rgb", "split": "test", "mode": "provided_context_generation", "cases": len(results),
        "limited_run": limit is not None, "dataset_sha256": fingerprint(prepared / "manifest.json"),
        "cases_sha256": fingerprint(prepared / "cases.jsonl"),
        "model": config.DEEPSEEK_MODEL, "temperature": config.DEEPSEEK_TEMPERATURE,
    })
    write_jsonl(output.with_name(output.stem + "_review_template.jsonl"), [review_template(row) for row in results])
    print(f"Results: {output}")
    return output


def review_template(result: dict) -> dict:
    return {
        "case_id": result["case_id"], "gold_evidence_verified": None,
        "no_answer_verified": None, "answer_correct": None,
        "action_correct": None, "citation_correct": None,
        "supported_claims": None, "unsupported_claims": None,
        "evidence_candidates": result.get("evidence_candidates", []),
        "verified_evidence": [],
        "reviewer": "", "review_notes": "",
    }


def merge_gold_reviews(result_path: Path, gold_path: Path) -> Path:
    gold = {item["case_id"]: item for item in rows(gold_path)}
    merged = []
    for result in rows(result_path):
        review = review_template(result)
        prior = gold.get(result["case_id"], {})
        for field in ("gold_evidence_verified", "no_answer_verified", "verified_evidence", "reviewer", "review_notes"):
            if field in prior:
                review[field] = prior[field]
        merged.append(review)
    output = result_path.with_name(result_path.stem + "_review_with_gold.jsonl")
    write_jsonl(output, merged)
    print(f"Merged {len(merged)} review rows -> {output}")
    return output


def mean(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def bootstrap_interval(values: list[float | None], *, seed: int = 20260926) -> list[float] | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    rng = random.Random(seed)
    sampled = sorted(sum(rng.choices(valid, k=len(valid))) / len(valid) for _ in range(1000))
    return [sampled[24], sampled[974]]


def summarize_retrieval(results: list[dict]) -> dict:
    groups = defaultdict(list)
    for result in results:
        groups[result["category"]].append(result)
    def metrics(items):
        source_linked = [row for row in items if row.get("gold_doc_ids")]
        coarse = [row["coarse_recall_at_20"] for row in source_linked]
        final = [row["final_recall_at_8"] for row in source_linked]
        return {"cases": len(items), "source_linked_cases": len(source_linked), "coarse_source_recall_at_20": mean(coarse), "coarse_recall_95pct_bootstrap": bootstrap_interval(coarse), "final_source_recall_at_8": mean(final), "final_recall_95pct_bootstrap": bootstrap_interval(final), "rerank_applied_cases": sum(bool(row["rerank_applied"]) for row in items)}
    track = results[0].get("track") if results else None
    note = ("CRUD 指标仅表示已关联来源文档的命中情况；具体证据片段及无答案边界仍待人工核实。" if track == "crud" else "MIRACL 使用原始人工相关性标注；未标注文段不视为负例。")
    note += " @20/@8 分别指实际检索流程的前 20/8 个切片，按其所属文档计算来源召回。"
    return {"overall": metrics(results), "by_category": {category: metrics(items) for category, items in groups.items()}, "note": note + " 未核实的问题不计入回答正确率、拒答率或幻觉率。"}


def write_retrieval_report(result_path: Path, results: list[dict]) -> None:
    summary = summarize_retrieval(results)
    write_json(result_path.with_name(result_path.stem + "_summary.json"), summary)
    def percent(value):
        return "待核实" if value is None else f"{value:.1%}"
    lines = [
        f"# {results[0]['track'].upper() if results else 'RAG'} 检索评测报告",
        "", f"样本数：{summary['overall']['cases']}；有来源标签的题：{summary['overall']['source_linked_cases']}。",
        "", "| 题型 | 题数 | 来源题数 | 粗排来源召回@20 | 重排来源召回@8 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, item in summary["by_category"].items():
        lines.append(f"| {category} | {item['cases']} | {item['source_linked_cases']} | {percent(item['coarse_source_recall_at_20'])} | {percent(item['final_source_recall_at_8'])} |")
    overall = summary["overall"]
    lines.extend([f"| **合计** | **{overall['cases']}** | **{overall['source_linked_cases']}** | **{percent(overall['coarse_source_recall_at_20'])}** | **{percent(overall['final_source_recall_at_8'])}** |", "", summary["note"], "", "抽样后的结果不能与公开完整基准的论文分数直接比较。JSON 内的 bootstrap 区间只反映该子集内的重抽样变化，不能消除选样偏差。", ""])
    result_path.with_name(result_path.stem + "_report.md").write_text("\n".join(lines), encoding="utf-8")


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def score_reviews(result_path: Path, review_path: Path) -> Path:
    results = {item["case_id"]: item for item in rows(result_path)}
    reviews = {item["case_id"]: item for item in rows(review_path)}
    if len(reviews) != len(rows(review_path)):
        raise ValueError("Duplicate case_id in review file")
    graded = []
    doc_dir = result_path.parent.parent / "prepared" / "documents"
    document_map_path = doc_dir.parent / "documents.jsonl"
    document_map = {item["doc_id"]: item["filename"] for item in rows(document_map_path)} if document_map_path.exists() else {}
    for case_id, result in results.items():
        review = reviews.get(case_id)
        if not review:
            continue
        gold_ready = review.get("no_answer_verified") is True if result["expected_action"] == "refuse" else review.get("gold_evidence_verified") is True
        fields = ("answer_correct", "action_correct", "citation_correct", "supported_claims", "unsupported_claims")
        if not gold_ready or any(review.get(field) is None for field in fields) or not review.get("reviewer", "").strip():
            continue
        if result["expected_action"] == "refuse" and not review.get("review_notes", "").strip():
            continue
        if result["expected_action"] != "refuse" and result.get("track") == "crud":
            verified = review.get("verified_evidence") or []
            if {item.get("doc_id") for item in verified} != set(result.get("gold_doc_ids", [])):
                continue
            for item in verified:
                filename = document_map.get(item["doc_id"])
                if not filename or not item.get("quote", "").strip() or item["quote"] not in (doc_dir / filename).read_text(encoding="utf-8"):
                    raise ValueError(f"Verified evidence is not verbatim in indexed source: {case_id}, {item['doc_id']}")
        if result.get("track") == "rgb" and result["expected_action"] != "refuse":
            verified = review.get("verified_evidence") or []
            if not verified or any(not item.get("quote", "").strip() or not any(item["quote"] in context for context in result.get("contexts", [])) for item in verified):
                continue
        if not isinstance(review["supported_claims"], int) or not isinstance(review["unsupported_claims"], int) or min(review["supported_claims"], review["unsupported_claims"]) < 0:
            raise ValueError(f"Invalid claim counts: {case_id}")
        if any(not isinstance(review[field], bool) for field in ("answer_correct", "action_correct", "citation_correct")):
            raise ValueError(f"Boolean review fields required: {case_id}")
        graded.append((result, review))
    categories = sorted({result["category"] for result in results.values()})
    def tally(pairs):
        n = len(pairs)
        correct = sum(review["answer_correct"] for _, review in pairs)
        action = sum(review["action_correct"] for _, review in pairs)
        supported = sum(review["supported_claims"] for _, review in pairs)
        unsupported = sum(review["unsupported_claims"] for _, review in pairs)
        claim_assessed = [(result, review) for result, review in pairs if review["supported_claims"] + review["unsupported_claims"] > 0]
        citation = sum(review["citation_correct"] for _, review in claim_assessed)
        with_unsupported = sum(review["unsupported_claims"] > 0 for _, review in claim_assessed)
        claims = supported + unsupported
        return {"reviewed": n, "claim_assessed_answers": len(claim_assessed), "answer_accuracy": correct / n if n else None, "answer_accuracy_95pct_wilson": wilson_interval(correct, n), "action_accuracy": action / n if n else None, "citation_accuracy": citation / len(claim_assessed) if claim_assessed else None, "unsupported_claim_rate": unsupported / claims if claims else None, "answer_with_unsupported_claim_rate": with_unsupported / len(claim_assessed) if claim_assessed else None, "claim_count": claims}
    breakdown = {
        "retrieval_miss": [result["case_id"] for result, _ in graded if result.get("gold_doc_ids") and result.get("final_recall_at_8") == 0],
        "wrong_action": [result["case_id"] for result, review in graded if not review["action_correct"]],
        "unsupported_generation": [result["case_id"] for result, review in graded if review["unsupported_claims"] > 0],
    }
    report = {"cases": len(results), "reviewed": len(graded), "overall": tally(graded), "by_category": {category: tally([(r, v) for r, v in graded if r["category"] == category]) for category in categories}, "error_breakdown": breakdown, "unreviewed_case_ids": [case_id for case_id in results if case_id not in {r["case_id"] for r, _ in graded}], "note": "Only human-verified gold evidence and completed answer reviews enter answer/factuality rates. Never interpret missing grades as zero errors."}
    output = result_path.with_name(result_path.stem + "_scored.json")
    write_json(output, report)
    def percent(value):
        return "待人工核实" if value is None else f"{value:.1%}"
    lines = [
        f"# {next(iter(results.values())).get('track', 'RAG').upper() if results else 'RAG'} 回答评测报告",
        "", f"生成回答：{len(results)} 题；已核实金标准并评审回答：{len(graded)} 题。",
        "", "| 题型 | 已评审 | 回答正确率 | 操作正确率 | 引用准确率 | 无依据事实占比 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category in [*categories, "合计"]:
        item = report["overall"] if category == "合计" else report["by_category"][category]
        lines.append(f"| {category} | {item['reviewed']} | {percent(item['answer_accuracy'])} | {percent(item['action_accuracy'])} | {percent(item['citation_accuracy'])} | {percent(item['unsupported_claim_rate'])} |")
    error_counts = [str(len(breakdown[key])) if graded else "待人工核实" for key in ("retrieval_miss", "wrong_action", "unsupported_generation")]
    lines.extend(["", "错误归因（已评审题）：检索遗漏 {}；回答或拒答动作错误 {}；无依据生成 {}。".format(*error_counts), "", "未完成原文证据与回答逐事实人工复核的题不计入正确率、拒答率或幻觉相关指标；待核实不等于 0%。", ""])
    output.with_name(result_path.stem + "_scored_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Reviewed {len(graded)}/{len(results)} cases -> {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("index", "run", "score", "report", "merge-gold"))
    parser.add_argument("track", nargs="?", choices=("crud", "miracl", "rgb"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--mode", choices=("retrieval", "e2e"), default="retrieval")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-ids-file", type=Path, help="One case ID per line; only cases in --split are allowed")
    parser.add_argument("--resume-results", type=Path, help="Resume a partial e2e JSONL result without repeating paid calls")
    parser.add_argument("--max-docs", type=int, help="Smoke index only; not usable for benchmark scores")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    if args.command == "index":
        if not args.track:
            parser.error("index needs a track")
        index_track(args.root, args.track, args.force, args.max_docs)
    elif args.command == "run":
        if not args.track:
            parser.error("run needs a track")
        if args.track == "rgb":
            if args.mode != "e2e":
                parser.error("RGB is a provided-context e2e track; use --mode e2e")
            rgb_run(args.root, args.limit)
        else:
            retrieval_run(args.root, args.track, args.split, args.limit, args.mode == "e2e", args.case_ids_file, args.resume_results)
    elif args.command == "score":
        if not args.results or not args.reviews:
            parser.error("score needs --results and --reviews")
        score_reviews(args.results, args.reviews)
    elif args.command == "report":
        if not args.results:
            parser.error("report needs --results")
        write_retrieval_report(args.results, rows(args.results))
        print(f"Report: {args.results.with_name(args.results.stem + '_report.md')}")
    else:
        if not args.results or not args.reviews:
            parser.error("merge-gold needs --results and --reviews (completed gold review file)")
        merge_gold_reviews(args.results, args.reviews)


if __name__ == "__main__":
    main()
