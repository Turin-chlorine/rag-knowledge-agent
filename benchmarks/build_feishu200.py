"""Build reviewable Feishu Base payloads for 110 legacy + 90 CRUD cases.

This script writes JSON payloads only. It never changes Feishu by itself.
Model-assisted labels remain clearly marked as provisional in every new row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from evaluate import rows, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data" / "benchmarks" / "feishu"


def percentage(part: int, total: int) -> str:
    return f"{part / total:.2%}（{part}/{total}）" if total else "待核实（0题可计分）"


def read_ndjson(path: Path) -> list[dict]:
    return rows(path)


def format_truth(case: dict, review: dict, docs: dict[str, dict]) -> str:
    gold = review.get("gold") or {}
    out = [f"上游参考答案：{case['reference_answer']}"]
    if review.get("status") == "model_review_complete":
        out.append(f"模型依入库原文核对的答案：{gold['verified_answer']}")
        out.append("逐字证据（程序已检查引文在对应原文中）：")
        for item in gold["evidence"]:
            info = docs[item["doc_id"]]
            out.append(f"[{item['doc_id']}] {item['quote']}")
            if info.get("source_url"):
                out.append(f"上游来源：{info['source_url']}")
    else:
        out.append("上游参考答案尚未得到有效原文证据核验，不能作为已确认真值。")
        if gold.get("notes"):
            out.append(f"待核实原因：{gold['notes']}")
    out.append("核验口径：模型辅助初评，逐字引文只证明文本存在；语义支持及事实判定仍待人工复核。")
    return "\n".join(out)


def format_sources(result: dict) -> str:
    out = []
    for source in result.get("sources", []):
        out.append(f"[{source['index']}] {source['doc_name']} / 片段 {source['chunk_index']} / 粗排 {source['retrieval_score']} / 重排 {source['rerank_score']}\n{source['content']}")
    return "\n\n".join(out)


def new_record(number: int, case: dict, result: dict, review: dict, docs: dict[str, dict]) -> dict:
    mapping = {"questanswer_1doc": "单跳事实题", "questanswer_2docs": "多文档对比题", "questanswer_3docs": "多文档对比题"}
    gold = set(case["gold_doc_ids"])
    found = gold & set(result.get("final_doc_ids", []))
    recall_label = "是" if found == gold else "部分" if found else "否"
    reviewed = review.get("status") == "model_review_complete"
    answer = review.get("answer_review") if reviewed else None
    source_names = list(dict.fromkeys(source["doc_name"] for source in result.get("sources", [])))
    record = {
        "题号": number, "问题": case["question"], "分类": [mapping[case["category"]]],
        "Agent最终回答": result["answer"], "原文真值(人工填写)": format_truth(case, review, docs),
        "检索来源文档": "+".join(source_names), "预期来源文档": "+".join(case["gold_doc_ids"]),
        "检索片段数": len(result.get("sources", [])),
        "检索原文出处(文档+片段全文+分数)": format_sources(result),
        "召回判定(人工填写)": recall_label,
        "评测集": "CRUD-RAG 中文新闻子集", "上游样本ID": case["case_id"], "数据拆分": case["split"],
        "真值核验状态": "模型辅助核验：逐字引文通过；待人工复核" if reviewed else "待核实：上游答案或模型引文未通过核验",
        "判定来源": "deepseek-chat（温度0）辅助初评 + 本地逐字引文校验；与答题同模型，存在自评偏差；非人工金标准",
    }
    if answer:
        relevant = set(answer["relevant_source_indices"])
        record["精确判定(人工填写)"] = f"{len(relevant) / len(result['sources']):.3f}" if result.get("sources") else "0"
        record["幻觉判定(人工填写)"] = "是" if any(not claim["supported"] for claim in answer["claims"]) else "否"
        record["任务完成判定(人工填写)"] = answer["task_completion"]
    return record


def build(args: argparse.Namespace) -> None:
    old = read_ndjson(args.original)
    selected = rows(args.selection)
    results = {row["case_id"]: row for path in (args.test_results, args.dev_results) for row in rows(path)}
    reviews = {row["case_id"]: row for row in rows(args.reviews)}
    docs = {row["doc_id"]: row for row in rows(ROOT / "data" / "benchmarks" / "crud" / "prepared" / "documents.jsonl")}
    if len(old) != 110 or sorted(row["题号"] for row in old) != list(range(1, 111)):
        raise ValueError("Legacy Feishu export is not exactly questions 1..110")
    if len(selected) != 90 or len({row["case_id"] for row in selected}) != 90 or any(row["case_id"] not in results or row["case_id"] not in reviews for row in selected):
        raise ValueError("Expected 90 unique selected cases with project answers and reviews")
    args.output.mkdir(parents=True, exist_ok=True)
    new = [new_record(number, case, results[case["case_id"]], reviews[case["case_id"]], docs) for number, case in enumerate(selected, 111)]
    for start in range(0, len(new), 15):
        write_json(args.output / f"new_batch_{start // 15 + 1:02}.json", {"create_records": new[start:start + 15]})
    legacy_updates = {row["record_id"]: {
        "评测集": "旧版8文档（历史回归）", "上游样本ID": f"legacy-{row['题号']:03}",
        "数据拆分": "legacy", "真值核验状态": "沿用原表既有标注；本轮未独立复核",
        "判定来源": "原表既有填写字段；未在本轮重新审计",
    } for row in old}
    write_json(args.output / "legacy_update.json", {"update_records": legacy_updates})

    reviewed = [(case, results[case["case_id"]], reviews[case["case_id"]]["answer_review"]) for case in selected if reviews[case["case_id"]].get("status") == "model_review_complete"]
    split_counts = Counter(case["split"] for case in selected)
    n = len(reviewed)
    complete = sum(answer["task_completion"] == "是" for _, _, answer in reviewed)
    partial = sum(answer["task_completion"] == "部分" for _, _, answer in reviewed)
    hallucinated = sum(any(not claim["supported"] for claim in answer["claims"]) for _, _, answer in reviewed)
    all_claims = [claim for _, _, answer in reviewed for claim in answer["claims"]]
    unsupported = sum(not claim["supported"] for claim in all_claims)
    relevant = sum(len(set(answer["relevant_source_indices"])) for _, _, answer in reviewed)
    source_chunks = sum(len(result.get("sources", [])) for _, result, _ in reviewed)
    source_pairs = sum(len(case["gold_doc_ids"]) for case in selected)
    found_pairs = sum(len(set(case["gold_doc_ids"]) & set(results[case["case_id"]].get("final_doc_ids", []))) for case in selected)
    all_sources_found = sum(set(case["gold_doc_ids"]) <= set(results[case["case_id"]].get("final_doc_ids", [])) for case in selected)
    test_reviewed = [(case, answer) for case, _, answer in reviewed if case["split"] == "test"]
    model_note = "新增CRUD-RAG子集；deepseek-chat温度0辅助判断，逐字引文经程序核对；答题与评审同模型，存在自评偏差，语义支持及评分待人工复核。初选8题在原文核验后被剔除并从同类dev候选补足，属于后验筛选。不可与旧版110题或公开完整基准分数直接合并。"
    def metric(name: str, value: str, note: str = model_note) -> dict:
        return {"指标项": name, "数值": value, "口径说明": note}
    summary = [
        metric("【200题总览】", "", "旧版8文档110题 + 新增CRUD-RAG中文新闻子集90题。"),
        metric("当前总题数", "200", "题目表题号1–200；旧版110题保留，新增90题各有稳定上游样本ID。"),
        metric("旧版历史回归题数", "110", "仍使用旧版8文档与原有标注，未重新运行或独立复核。"),
        metric("新增公开语料题数", "90", f"单文档30、双文档30、三文档30；dev {split_counts['dev']}、test {split_counts['test']}；均通过项目真实检索和DeepSeek回答。"),
        metric("初选未通过自动核验题数", "8", "初选90题中8题因上游答案缺乏入库原文支持或模型引文无法逐字核对而剔除；这不等于8题均被证实为错题。从同题型dev候选补足，保留本地逐题审计记录。"),
        metric("跨语料统一幻觉率", "不计算", "旧版与新增题的文档、抽样和判定来源不同；不可把旧版0%扩展为200题结果。"),
        metric("【新增90题：模型辅助初评】", "", model_note),
        metric("模型评审独立性", "不独立", "项目回答和证据/事实初评均使用deepseek-chat；可能有同模型自评偏差，需独立人工复核或异构评审模型后才能作为最终质量指标。"),
        metric("原文证据自动核验通过题数", f"{n}/90", "模型判断上游答案得到入库原文支持，且给出的逐字引文在对应文档中存在；仍待人工检查语义支持。"),
        metric("来源文档召回@8", percentage(found_pairs, source_pairs), "按上游关联来源文档ID计算：最终8个切片覆盖的来源文档数/全部预期来源文档数；不等于片段证据召回。"),
        metric("全部来源命中题率@8", percentage(all_sources_found, 90), "90题中所有上游来源文档均进入最终8个切片的题数/90。"),
        metric("片段精确率（AI初评）", percentage(relevant, source_chunks), "模型判定直接有助作答的最终检索切片数/已初评题的最终检索切片数；非人工精确率。"),
        metric("回答完全完成率（AI初评）", percentage(complete, n), "模型初评任务完成判定为“是”的题数/已初评题数。"),
        metric("回答部分完成率（AI初评）", percentage(partial, n), "模型初评任务完成判定为“部分”的题数/已初评题数。"),
        metric("含无依据事实回答率（AI初评）", percentage(hallucinated, n), "模型初评至少一条回答事实缺少实际检索片段证据的题数/已初评题数；尚非人工幻觉率。"),
        metric("无依据事实陈述占比（AI初评）", percentage(unsupported, len(all_claims)), "模型初评无依据原子事实数/所有已拆分原子事实数。"),
        metric("引用准确题率（AI初评）", percentage(sum(answer["citation_correct"] for _, _, answer in reviewed), n), "模型初评回答中的[n]对应事实来源正确的题数/已初评题数。"),
        metric("保留test题完全完成率（AI初评）", percentage(sum(answer["task_completion"] == "是" for _, answer in test_reviewed), len(test_reviewed)), f"新增CRUD初选test 30题中有2题后验剔除，保留{split_counts['test']}题；此数仅作诊断，不能作为未经筛选的独立测试成绩。"),
    ]
    write_json(args.output / "summary_create.json", {"create_records": summary})
    previous = read_ndjson(args.original_summary)
    updates = {}
    for row in previous:
        name = row["指标项"]
        if name == "评测规模":
            updates[row["record_id"]] = {"指标项": "旧版评测规模（110题）", "口径说明": "仅旧版8文档历史回归；不包含新增90题。"}
        elif name in ("【检索层指标】", "【端到端指标】", "【四大指标公式与计算】", "【精确判定小数的计算说明】"):
            updates[row["record_id"]] = {"指标项": name.replace("【", "【旧版110题", 1)}
        elif name == "幻觉率":
            updates[row["record_id"]] = {"口径说明": "仅旧版110题原表历史标注0/110；本轮未独立复核，不能代表新增90题或200题整体的幻觉率。"}
    write_json(args.output / "summary_update.json", {"update_records": updates})
    write_json(args.output / "build_manifest.json", {
        "legacy_count": len(old), "new_count": len(new), "model_review_complete": n,
        "review_statuses": dict(Counter(reviews[case["case_id"]]["status"] for case in selected)),
        "source_result_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.test_results, args.dev_results, args.reviews)},
        "payload_batches": 6, "summary_rows_to_create": len(summary), "summary_rows_to_update": len(updates),
        "warning": "Automated judgments are provisional; no model verdict is labeled human gold.",
    })
    print(f"Built {len(new)} new rows in 6 batches; model-reviewed {n}/90; summary rows {len(summary)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT / "original110.ndjson")
    parser.add_argument("--original-summary", type=Path, default=DEFAULT / "original_summary.ndjson")
    parser.add_argument("--selection", type=Path, default=DEFAULT / "selected90.jsonl")
    parser.add_argument("--test-results", type=Path, required=True)
    parser.add_argument("--dev-results", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT / "payloads")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
