"""Prepare a full in-place replacement of the legacy 110 Feishu questions.

The resulting payloads update existing record IDs and never delete the table.
They are inert until submitted to Feishu by lark-cli.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_feishu200 import format_sources, format_truth, new_record, percentage
from evaluate import rows, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "benchmarks"
FEISHU = BASE / "feishu"
MODEL_NOTE = "deepseek-chat 温度0辅助初评，回答与评审使用同一模型，存在自评偏差；支持引文只经逐字存在性校验，语义判定待人工复核。"


def as_map(*paths: Path) -> dict[str, dict]:
    return {row["case_id"]: row for path in paths for row in rows(path)}


def display_truth(case: dict, review: dict, source_type: str) -> str:
    action = case["expected_action"]
    reference = json.dumps(case.get("reference_answer"), ensure_ascii=False)
    lines = [f"上游参考答案：{reference}", f"数据集预期操作：{action}"]
    if source_type == "rgb":
        lines.append("证据范围：本题给定的 RGB 片段；这些片段不是项目检索所得。")
        contexts = ["\n".join(x) if isinstance(x, list) else x for x in case["contexts"]]
        def strings(value):
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                return [part for item in value for part in strings(item)]
            return []
        targets = strings(case.get("reference_answer"))
        for target in targets:
            if not isinstance(target, str) or not target.strip():
                continue
            match = next(((i, text.find(target), text) for i, text in enumerate(contexts, 1) if target in text), None)
            if match:
                index, start, text = match
                lines.append(f"上游答案逐字定位 [{index}]：{text[max(0, start - 65):min(len(text), start + len(target) + 65)]}")
            else:
                lines.append(f"上游答案未在给定片段中逐字找到：{target}")
    else:
        lines.append("证据范围：本次入库的 400 篇 CRUD-RAG 文档；该题为近邻无答案候选，未有人工作全库无答案核验。")
    verdict = review.get("answer_review") or {}
    if verdict:
        lines.append(f"模型辅助判断标签可由给定证据支持：{'是' if verdict.get('label_supported') else '否'}；说明：{verdict.get('notes', '')}")
    else:
        lines.append("模型初评未完成，标签尚未复核。")
    lines.append("以上是上游标签与模型辅助核验，非人工确认真值；不能用上游参考答案替代文档证据。")
    return "\n".join(lines)


def special_record(number: int, case: dict, result: dict, review: dict) -> dict:
    rgb = case["case_id"].startswith("rgb-")
    answer = review.get("answer_review") if review.get("status") == "model_review_complete" else None
    if rgb:
        contexts = ["\n".join(x) if isinstance(x, list) else x for x in case["contexts"]]
        sources = "\n\n".join(f"[给定片段 {i}]\n{v}" for i, v in enumerate(contexts, 1))
        category = {"noise_rejection": "库内无答案题", "integration": "多文档对比题", "counterfactual": "易混淆题"}[case["category"]]
        dataset = "RGB 中文给定片段专项"
        retrieved = "RGB 给定片段（无检索）"
        expected = "RGB 给定片段"
        recall = "不适用：给定片段，未运行检索"
        count = len(contexts)
    else:
        sources = format_sources(result)
        category = "库内无答案题"
        dataset = "CRUD-RAG 中文新闻子集"
        retrieved = "+".join(dict.fromkeys(x["doc_name"] for x in result.get("sources", [])))
        expected = "全库无答案候选，未人工确认"
        recall = "不适用：无正例来源文档"
        count = len(result.get("sources", []))
    record = {
        "题号": number, "问题": case["question"], "分类": [category],
        "Agent最终回答": result["answer"],
        "原文真值(人工填写)": display_truth(case, review, "rgb" if rgb else "crud"),
        "检索来源文档": retrieved, "预期来源文档": expected,
        "检索片段数": count, "检索原文出处(文档+片段全文+分数)": sources,
        "召回判定(人工填写)": recall, "评测集": dataset,
        "上游样本ID": case["case_id"], "数据拆分": case["split"],
        "真值核验状态": "模型辅助标签核验；待人工复核" if answer else "模型核验失败；待人工复核",
        "判定来源": MODEL_NOTE,
        "精确判定(人工填写)": None, "幻觉判定(人工填写)": None,
        "任务完成判定(人工填写)": None,
    }
    if answer:
        record["精确判定(人工填写)"] = f"{len(set(answer['relevant_source_indices'])) / count:.3f}" if count else "0"
        record["幻觉判定(人工填写)"] = "是" if any(not x["supported"] for x in answer["claims"]) else "否"
        record["任务完成判定(人工填写)"] = answer["task_completion"]
    return record


def metric(name: str, value: str, note: str = MODEL_NOTE) -> dict:
    return {"指标项": name, "数值": value, "口径说明": note}


def review_stats(items: list[tuple[dict, dict, dict]]) -> list[dict]:
    valid = [(case, result, rev["answer_review"]) for case, result, rev in items if rev.get("status") == "model_review_complete" and rev.get("answer_review")]
    n = len(valid)
    claims = [claim for _, _, review in valid for claim in review["claims"]]
    unsupported = sum(not claim["supported"] for claim in claims)
    return [
        metric("题数", str(len(items)), "固定选题；每题对应稳定上游样本 ID。"),
        metric("可参与模型初评题数", f"{n}/{len(items)}", "核验失败题不进入下列回答指标分母。"),
        metric("标签证据核验通过", f"{sum((rev.get('gold') or {}).get('gold_supported', False) if case['case_id'].startswith('crud-questanswer') else (rev.get('answer_review') or {}).get('label_supported', False) for case, _, rev in items)}/{len(items)}", "CRUD 答案题要求模型提供入库原文逐字引文；无答案和 RGB 为模型判断，缺少人工全量复核。"),
        metric("操作正确率（模型初评）", percentage(sum(review["action_correct"] for _, _, review in valid), n)),
        metric("回答正确率（模型初评）", percentage(sum(review["answer_correct"] for _, _, review in valid), n)),
        metric("含无依据事实回答率（模型初评）", percentage(sum(any(not claim["supported"] for claim in review["claims"]) for _, _, review in valid), n), "回答中至少一条原子事实未找到给定证据的题数/已评审题数；该数为模型初评，不是人工确认幻觉率。"),
        metric("无依据事实陈述占比（模型初评）", percentage(unsupported, len(claims)), "无依据原子事实数/全部模型拆分事实数；拒答若没有事实陈述则不增加分母。"),
        metric("引用准确题率（模型初评）", percentage(sum(review["citation_correct"] for _, _, review in valid), n)),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("crud-dev43", "crud-test7", "rgb-results", "crud30-reviews", "crud20-reviews", "rgb60-reviews"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--output", type=Path, default=FEISHU / "replacement200")
    args = p.parse_args()
    old = rows(FEISHU / "final200_check.ndjson")
    old_by_number = {int(x["题号"]): x for x in old}
    if len(old) != 200 or len(old_by_number) != 200:
        raise ValueError("The verified Feishu snapshot must have exactly 200 numbered records")
    selected90 = rows(FEISHU / "selected90.jsonl")
    crud_cases = rows(BASE / "crud" / "prepared" / "cases.jsonl")
    selected_ids = {x["case_id"] for x in selected90}
    remaining = [x for x in crud_cases if x["case_id"] not in selected_ids]
    rgb_cases = rows(BASE / "rgb" / "prepared" / "cases.jsonl")
    if (len(selected90), len(remaining), len(rgb_cases)) != (90, 50, 60):
        raise ValueError("Expected fixed 90+50+60 case selection")
    new_cases = remaining + rgb_cases
    if len({x["case_id"] for x in selected90 + new_cases}) != 200:
        raise ValueError("Case IDs are not unique")
    results = as_map(args.crud_dev43, args.crud_test7, args.rgb_results)
    reviews = as_map(args.crud30_reviews, args.crud20_reviews, args.rgb60_reviews)
    reviews.update(as_map(FEISHU / "final90_reviews.jsonl"))
    old_answers = as_map(BASE / "crud" / "runs" / "test_e2e_20260925T173930Z.jsonl", BASE / "crud" / "runs" / "dev_e2e_selected60_20260926T042955Z.jsonl", BASE / "crud" / "runs" / "dev_e2e_selected10_20260926T045420Z.jsonl")
    old_answers.update(as_map(FEISHU / "final90_answers.jsonl"))
    results.update(old_answers)
    missing = {x["case_id"] for x in selected90 + new_cases} - (set(results) & set(reviews))
    if missing:
        raise ValueError(f"Missing results or reviews: {sorted(missing)[:8]}")
    docs = {x["doc_id"]: x for x in rows(BASE / "crud" / "prepared" / "documents.jsonl")}
    updates = {}
    for number, case in enumerate(new_cases, 1):
        result, review = results[case["case_id"]], reviews[case["case_id"]]
        if case["category"] == "near_no_answer" or case["case_id"].startswith("rgb-"):
            record = special_record(number, case, result, review)
        else:
            record = new_record(number, case, result, review, docs)
            for field in ("精确判定(人工填写)", "幻觉判定(人工填写)", "任务完成判定(人工填写)"):
                record.setdefault(field, None)
        updates[old_by_number[number]["record_id"]] = record
    args.output.mkdir(parents=True, exist_ok=True)
    update_items = list(updates.items())
    for start in range(0, len(update_items), 10):
        write_json(args.output / f"questions_{start // 10 + 1:02}.json", {"update_records": dict(update_items[start:start + 10])})

    groups = {
        "CRUD 可回答题（完整检索链）": [x for x in selected90 + remaining if x["category"] != "near_no_answer"],
        "CRUD 库内无答案候选（完整检索链）": [x for x in remaining if x["category"] == "near_no_answer"],
        "RGB 噪声拒答（给定片段）": [x for x in rgb_cases if x["category"] == "noise_rejection"],
        "RGB 多片段整合（给定片段）": [x for x in rgb_cases if x["category"] == "integration"],
        "RGB 反事实冲突（给定片段）": [x for x in rgb_cases if x["category"] == "counterfactual"],
    }
    all_items = [(x, results[x["case_id"]], reviews[x["case_id"]]) for x in selected90 + new_cases]
    valid = [(x, result, review["answer_review"]) for x, result, review in all_items if review.get("status") == "model_review_complete" and review.get("answer_review")]
    n = len(valid)
    hallucinated = sum(any(not c["supported"] for c in review["claims"]) for _, _, review in valid)
    all_claims = [c for _, _, review in valid for c in review["claims"]]
    crud_answerables = groups["CRUD 可回答题（完整检索链）"]
    source_pairs = sum(len(x["gold_doc_ids"]) for x in crud_answerables)
    found_pairs = sum(len(set(x["gold_doc_ids"]) & set(results[x["case_id"]]["final_doc_ids"])) for x in crud_answerables)
    summary = [
        metric("【全新公开语料200题总览】", "", "旧版8文档110题已在原记录位置全部替换；题号1–200均为公开语料题。"),
        metric("总题数", "200", "CRUD-RAG 140题，RGB中文60题；原始110题不再参与统计。"),
        metric("CRUD 完整检索链题数", "140", "400篇固定抽样中文新闻文档入库；120题可回答，20题近邻无答案候选。"),
        metric("RGB 给定片段专项题数", "60", "噪声拒答20、多片段整合20、反事实冲突20；只运行项目生成链，未运行项目检索。"),
        metric("项目运行完成题数", "200/200", "CRUD题通过真实向量检索、重排与生成；RGB题通过项目给定片段生成路径。"),
        metric("模型辅助判定覆盖", f"{n}/200", "失败或未通过证据核验的题不进入回答指标分母。"),
        metric("人工确认幻觉率", "待人工复核", "现有判定来自与答题相同的DeepSeek模型；不能声称人工确认。"),
        metric("含无依据事实回答率（模型初评，全200题）", percentage(hallucinated, n), "至少一条回答事实没有给定检索片段或RGB片段支持的题数/有效模型评审题数；包括拒答题，按同模型初评，存在偏差。"),
        metric("无依据事实陈述占比（模型初评，全200题）", percentage(sum(not c["supported"] for c in all_claims), len(all_claims)), "模型拆分并判为无依据的原子事实/模型拆分全部原子事实。"),
        metric("回答正确率（模型初评，全200题）", percentage(sum(review["answer_correct"] for _, _, review in valid), n)),
        metric("操作正确率（模型初评，全200题）", percentage(sum(review["action_correct"] for _, _, review in valid), n)),
        metric("任务完全完成率（模型初评，全200题）", percentage(sum(review["task_completion"] == "是" for _, _, review in valid), n)),
        metric("CRUD 来源文档召回@8", percentage(found_pairs, source_pairs), "只统计120道 CRUD 可回答题：最终8片段命中的上游来源文档数/全部上游来源文档数；不等于证据片段召回。"),
        metric("可回答题原文证据核验通过", f"{sum(reviews[x['case_id']].get('status') == 'model_review_complete' for x in crud_answerables)}/120", "模型提供的答案引文经程序检查逐字存在；语义支持待人工复核。"),
        metric("数据集与评审局限", "模型初评，非公开基准分数", "CRUD固定140题已全部纳入，先前90题阶段剔除的8题也重新进入本次200题并标示核验状态；先前题目检查使用过test，独立性受限；不能与论文完整基准直接比较。"),
        metric("【分轨指标】", "", "下列五组分别统计；RGB不与CRUD合并计算检索指标。"),
    ]
    for name, cases in groups.items():
        summary.extend(metric(f"{name}｜{item['指标项']}", item["数值"], item["口径说明"]) for item in review_stats([(x, results[x["case_id"]], reviews[x["case_id"]]) for x in cases]))
    if len(summary) != 56:
        raise ValueError(f"Expected 56 replacement summary rows, got {len(summary)}")
    existing_summary = rows(FEISHU / "final_summary56_check.ndjson")
    if len(existing_summary) != 56:
        raise ValueError("Expected 56 existing summary records")
    summary_updates = {row["record_id"]: value for row, value in zip(existing_summary, summary)}
    for start in range(0, 56, 14):
        write_json(args.output / f"summary_{start // 14 + 1:02}.json", {"update_records": dict(list(summary_updates.items())[start:start + 14])})
    write_json(args.output / "manifest.json", {
        "questions": 200, "replaced_legacy": 110, "summary_rows": 56,
        "groups": {k: len(v) for k, v in groups.items()},
        "reviewed": n, "unsupported_answer_cases": hallucinated,
        "old_record_ids_sha256": hashlib.sha256("\n".join(updates).encode()).hexdigest(),
        "warning": "Model-assisted provisional scoring; neither gold labels nor factual entailment are fully human-verified.",
    })
    print(f"Prepared 110 question replacements and 56 summary replacements; reviewed {n}/200")


if __name__ == "__main__":
    main()
