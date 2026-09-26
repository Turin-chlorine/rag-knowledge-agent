"""Build Feishu replacement payloads for 200 CRUD cases run through upload/chat API."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_feishu200 import new_record, percentage
from evaluate import rows, write_json
from replace_feishu200 import special_record

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "benchmarks" / "upload200" / "prepared"
RUN = ROOT / "data" / "benchmarks" / "upload200" / "api_run"
PRIOR = ROOT / "data" / "benchmarks" / "feishu" / "replacement200"
OUT = RUN / "feishu_payloads"
DATASET = "CRUD-RAG 真实上传接口评测"
PROVENANCE = "项目 POST /api/documents/upload + POST /api/chat；DeepSeek 同模型辅助初评与逐字引文校验，语义支持待人工复核"


def metric(name: str, value: str, note: str) -> dict:
    return {"指标项": name, "数值": value, "口径说明": note}


def annotated(items: list[tuple[dict, dict, dict]]) -> list[tuple[dict, dict, dict]]:
    return [(c, r, v["answer_review"]) for c, r, v in items
            if v.get("status") == "model_review_complete" and v.get("answer_review")
            and (c["expected_action"] == "answer" or v["answer_review"].get("label_supported") is True)]


def group_metrics(title: str, items: list[tuple[dict, dict, dict]]) -> list[dict]:
    reviewed = annotated(items)
    n = len(reviewed)
    claims = [claim for _, _, v in reviewed for claim in v["claims"]]
    answerable = [c for c, _, _ in items if c["expected_action"] == "answer"]
    gold_pairs = sum(len(c["gold_doc_ids"]) for c in answerable)
    result_by_id = {c["case_id"]: r for c, r, _ in items}
    found_pairs = sum(len(set(c["gold_doc_ids"]) & set(result_by_id[c["case_id"]]["final_doc_ids"])) for c in answerable)
    gold_ok = sum(v.get("status") == "model_review_complete" if c["expected_action"] == "answer" else (v.get("answer_review") or {}).get("label_supported", False) for c, _, v in items)
    note = "本组答案与评审均使用 DeepSeek；回答事实需人工复核。"
    return [
        metric(title + "｜题数", str(len(items)), "固定 CRUD-RAG 样本，全部通过上传接口与问答接口。"),
        metric(title + "｜模型初评有效题数", f"{n}/{len(items)}", "原文标签核验或引文校验失败的题不计入下列回答指标。"),
        metric(title + "｜标签证据初核通过", f"{gold_ok}/{len(items)}", "可回答题要求模型给出入库原文逐字引文；拒答题仅为模型初核，未有人工作全库确认。"),
        metric(title + "｜来源文档召回@8", percentage(found_pairs, gold_pairs) if gold_pairs else "不适用：无正例来源文档", "最终检索前8切片命中的上游来源文档数/全部来源文档数；不是证据片段召回。"),
        metric(title + "｜回答正确率（模型初评）", percentage(sum(v["answer_correct"] for _, _, v in reviewed), n), note),
        metric(title + "｜操作正确率（模型初评）", percentage(sum(v["action_correct"] for _, _, v in reviewed), n), note),
        metric(title + "｜完全完成率（模型初评）", percentage(sum(v["task_completion"] == "是" for _, _, v in reviewed), n), note),
        metric(title + "｜含无依据事实回答率（模型初评）", percentage(sum(any(not claim["supported"] for claim in v["claims"]) for _, _, v in reviewed), n), "至少一条回答事实无实际检索片段支持的题数/有效初评题数；人工幻觉率另列待核实。"),
        metric(title + "｜无依据事实陈述占比（模型初评）", percentage(sum(not claim["supported"] for claim in claims), len(claims)), "无依据原子事实数/模型拆分的全部原子事实数。"),
    ]


def main() -> None:
    cases = rows(PREPARED / "cases.jsonl")
    results = {x["case_id"]: x for x in rows(RUN / "crud200_api_answers.jsonl")}
    reviews = {x["case_id"]: x for x in rows(RUN / "answerable180_reviews.jsonl") + rows(RUN / "negative20_reviews.jsonl")}
    docs = {x["doc_id"]: x for x in rows(PREPARED / "documents.jsonl")}
    old_questions = rows(PRIOR / "verified_questions.ndjson")
    old_summary = rows(PRIOR / "verified_summary.ndjson")
    if len(cases) != 200 or len(results) != 200 or len(reviews) != 200 or len(old_questions) != 200 or len(old_summary) != 56:
        raise ValueError("Expected 200 cases/results/reviews/Feishu rows and 56 summary rows")
    if {c["case_id"] for c in cases} != set(results) or set(results) != set(reviews):
        raise ValueError("Case, result, and review IDs differ")
    old_by_number = {int(x["题号"]): x for x in old_questions}
    if sorted(old_by_number) != list(range(1, 201)):
        raise ValueError("Existing Feishu numbering is not 1..200")
    OUT.mkdir(parents=True, exist_ok=True)
    updates = {}
    for number, case in enumerate(cases, 1):
        result, review = results[case["case_id"]], reviews[case["case_id"]]
        if case["expected_action"] == "answer":
            value = new_record(number, case, result, review, docs)
            for field in ("精确判定(人工填写)", "幻觉判定(人工填写)", "任务完成判定(人工填写)"):
                value.setdefault(field, None)
        else:
            value = special_record(number, case, result, review)
            value["原文真值(人工填写)"] = (
                f"上游参考答案（本题应拒答，不能作为入库证据）：{case['reference_answer']}\n"
                "预期操作：refuse。现有 520 篇入库文档未经过人工逐篇无答案核验；"
                "这是近邻无答案候选，不能作为已确认拒答金标准。\n"
                f"模型初评说明：{(review.get('answer_review') or {}).get('notes', '未完成')}"
            )
        value["评测集"] = DATASET
        value["题型细分"] = {
            "questanswer_1doc": "单文档事实题",
            "questanswer_2docs": "双文档综合题",
            "questanswer_3docs": "三文档综合题",
            "near_no_answer": "近邻无答案候选",
        }[case["category"]]
        value["判定来源"] = PROVENANCE
        updates[old_by_number[number]["record_id"]] = value
    records = list(updates.items())
    for start in range(0, 200, 10):
        write_json(OUT / f"questions_{start // 10 + 1:02}.json", {"update_records": dict(records[start:start + 10])})
    upload = json.loads((RUN / "upload_verification.json").read_text(encoding="utf-8"))
    all_items = [(c, results[c["case_id"]], reviews[c["case_id"]]) for c in cases]
    valid = annotated(all_items)
    n = len(valid)
    claims = [claim for _, _, v in valid for claim in v["claims"]]
    hallucinated = sum(any(not claim["supported"] for claim in v["claims"]) for _, _, v in valid)
    answerable = [c for c in cases if c["expected_action"] == "answer"]
    gold_pairs = sum(len(c["gold_doc_ids"]) for c in answerable)
    found_pairs = sum(len(set(c["gold_doc_ids"]) & set(results[c["case_id"]]["final_doc_ids"])) for c in answerable)
    groups = {
        "单文档事实题": [item for item in all_items if item[0]["category"] == "questanswer_1doc"],
        "双文档综合题": [item for item in all_items if item[0]["category"] == "questanswer_2docs"],
        "三文档综合题": [item for item in all_items if item[0]["category"] == "questanswer_3docs"],
        "库内无答案候选": [item for item in all_items if item[0]["category"] == "near_no_answer"],
    }
    overview = [
        metric("【CRUD-RAG 真实上传200题】", "", "200题全部来自 CRUD-RAG；RGB不再参与。"),
        metric("评测总题数", "200", "原有 200 行已全部换为 CRUD-RAG 题，稳定上游样本ID去重。"),
        metric("文档上传数", str(upload["document_count"]), "桌面 TXT 通过项目 POST /api/documents/upload 逐篇导入隔离知识库。"),
        metric("项目切片数", str(upload["chunk_count"]), "上传接口调用项目解析、切片、Embedding、JSON向量存储。"),
        metric("项目回答完成数", "200/200", "每题独立会话，通过项目 POST /api/chat，运行检索、重排及生成。"),
        metric("单文档题数", "60", "CRUD 上游单来源问答。"),
        metric("双文档题数", "60", "CRUD 上游双来源问答。"),
        metric("三文档题数", "60", "CRUD 上游三来源问答。"),
        metric("近邻无答案候选题数", "20", "未完成520篇全库人工无答案核查，不视作确定负例。"),
        metric("模型辅助有效评审题数", f"{n}/200", "未通过真值引文或事实引文校验的题不进入回答指标分母。"),
        metric("人工确认幻觉率", "待人工复核", "同一 DeepSeek 模型生成和初评，逐字引文存在不等于人工语义支持判定。"),
        metric("含无依据事实回答率（模型初评）", percentage(hallucinated, n), "至少一条回答事实缺少实际检索片段支持的题数/模型有效评审题数。"),
        metric("无依据事实陈述占比（模型初评）", percentage(sum(not x["supported"] for x in claims), len(claims)), "模型拆分出的无依据原子事实数/全部原子事实数。"),
        metric("回答正确率（模型初评）", percentage(sum(v["answer_correct"] for _, _, v in valid), n), "同模型初评，人工真值核验待完成。"),
        metric("操作正确率（模型初评）", percentage(sum(v["action_correct"] for _, _, v in valid), n), "回答或拒答动作正确题数/有效评审题数。"),
        metric("完全完成率（模型初评）", percentage(sum(v["task_completion"] == "是" for _, _, v in valid), n), "任务完成标签来自模型初评。"),
        metric("CRUD 来源文档召回@8", percentage(found_pairs, gold_pairs), "180道可回答题，最终8切片命中的上游来源文档数/上游来源文档总数。"),
        metric("可回答题真值初核通过", f"{sum(reviews[c['case_id']].get('status') == 'model_review_complete' for c in answerable)}/180", "模型提供入库原文逐字引文；语义支持待人工核对。"),
        metric("检索方式", "项目真实上传+问答API", "与网页拖入同一上传后端路径；使用隔离知识库，不混入日常文档。"),
        metric("数据与计分边界", "固定子集，模型初评", "文档是 CRUD 上游 news 字段原文片段，部分不足以证明改写参考答案；未核验负例与失败标签不当作0错误。"),
    ]
    summary = overview + [row for title, items in groups.items() for row in group_metrics(title, items)]
    if len(summary) != 56:
        raise ValueError(f"Expected 56 summary rows, got {len(summary)}")
    summary_updates = {old["record_id"]: value for old, value in zip(old_summary, summary)}
    parts = list(summary_updates.items())
    for start in range(0, 56, 14):
        write_json(OUT / f"summary_{start // 14 + 1:02}.json", {"update_records": dict(parts[start:start + 14])})
    write_json(OUT / "manifest.json", {
        "case_count": 200, "document_count": upload["document_count"], "chunk_count": upload["chunk_count"],
        "reviewed": n, "unsupported_answer_cases": hallucinated,
        "review_statuses": dict(Counter(x["status"] for x in reviews.values())),
        "group_counts": {title: len(items) for title, items in groups.items()},
        "summary_rows": len(summary), "warning": "All answer quality judgments are provisional model-assisted reviews",
    })
    print(f"Prepared 200 question and 56 summary replacements; reviewed {n}, unsupported answers {hallucinated}")


if __name__ == "__main__":
    main()
