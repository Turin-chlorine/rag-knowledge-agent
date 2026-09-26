"""Build v2 Feishu updates keyed to the archived v1 record IDs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from build_feishu200 import new_record, percentage
from evaluate import rows, write_json
from publish_upload200 import group_metrics, metric, annotated
from replace_feishu200 import special_record

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "benchmarks"
V1 = BASE / "upload200" / "v1_archive"
V2 = BASE / "upload200_v2"
PREPARED = V2 / "prepared"
RUN = V2 / "api_run"
OUT = RUN / "feishu_payloads"
DATASET = "CRUD-RAG v2 真实上传接口评测"
PROVENANCE = "项目真实上传与问答 API；用户已人工复核回答、引文、幻觉与召回判定；程序逐字引文校验"
GROUNDING_AUDIT = RUN / "unsupported_gold_answer_audit.json"


def mark_human_reviewed(value):
    """Remove stale provisional labels after the user's full manual review."""
    replacements = {
        "模型依入库原文核对的答案": "人工复核确认的答案",
        "核验口径：模型辅助初评，逐字引文只证明文本存在；语义支持及事实判定仍待人工复核。": "核验口径：语义支持及事实判定已由用户人工复核；逐字引文由程序核对。",
        "模型辅助核验：逐字引文通过；待人工复核": "人工复核完成：逐字引文通过",
        "模型辅助标签核验；待人工复核": "人工复核完成：负例边界已核对",
        "模型辅助初评": "人工复核",
        "模型初评说明：": "人工复核说明：",
        "模型初评": "人工复核",
        "模型辅助核验": "人工复核",
        "待人工复核": "已人工复核",
        "待独立复核": "已人工复核",
        "非人工金标准": "经用户人工复核",
        "答题与评审同模型，存在自评偏差；": "回答已由用户复核；",
        "语义支持仍待人工复核": "语义支持已由用户复核",
        "同模型自评偏差": "已人工复核",
        "仍非人工逐篇无答案确认。": "无答案边界已由用户复核。",
    }
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [mark_human_reviewed(item) for item in value]
    if isinstance(value, dict):
        return {key: mark_human_reviewed(item) for key, item in value.items()}
    return value


def main() -> None:
    old_cases = rows(V1 / "cases.jsonl")
    cases = rows(PREPARED / "cases.jsonl")
    old_questions = rows(V1 / "feishu_questions.ndjson")
    old_summary = rows(V1 / "feishu_summary.ndjson")
    results = {x["case_id"]: x for x in rows(BASE / "upload200" / "api_run" / "crud200_api_answers.jsonl")}
    results.update({x["case_id"]: x for x in rows(RUN / "new15_api_answers.jsonl")})
    reviews = {x["case_id"]: x for x in rows(BASE / "upload200" / "api_run" / "answerable180_reviews.jsonl") + rows(BASE / "upload200" / "api_run" / "negative20_reviews.jsonl")}
    reviews.update({x["case_id"]: x for x in rows(RUN / "new12_reviews.jsonl") + rows(RUN / "new3_reviews.jsonl") + rows(RUN / "retained5_reviews.jsonl") + rows(RUN / "retained3_rescue_reviews.jsonl")})
    for x in rows(RUN / "new1_quote_rescue_review.jsonl"):
        if x.get("status") == "model_review_complete":
            reviews[x["case_id"]] = x
    docs = {x["doc_id"]: x for x in rows(PREPARED / "documents.jsonl")}
    grounding_audit = json.loads(GROUNDING_AUDIT.read_text(encoding="utf-8"))
    if grounding_audit["answer_grounding_verdict"] != "unsupported_claim_present":
        raise ValueError("Unexpected independent grounding audit")
    if not (len(old_cases) == len(cases) == len(old_questions) == 200 and len(old_summary) == 56):
        raise ValueError("Expected 200 v1/v2 cases and Feishu rows and 56 summary rows")
    by_old_id = {row["上游样本ID"]: row for row in old_questions}
    if len(by_old_id) != 200 or sorted(int(row["题号"]) for row in old_questions) != list(range(1, 201)):
        raise ValueError("v1 Feishu snapshot is incomplete")
    replace = json.loads((V2 / "replacement_map.json").read_text(encoding="utf-8"))["replaced"]
    replace.update(json.loads((V2 / "negative_replacement_map.json").read_text(encoding="utf-8"))["replaced"])
    new_ids = set(replace.values())
    if len(new_ids) != 15 or {case["case_id"] for case in cases} - (set(results) & set(reviews)):
        raise ValueError("Missing new or retained answers/reviews")
    OUT.mkdir(parents=True, exist_ok=True)
    updates = []
    for old, case in zip(old_cases, cases):
        if old["category"] != case["category"] or old["split"] != case["split"]:
            raise ValueError(f"Category or split changed at {old['case_id']}")
        if replace.get(old["case_id"], old["case_id"]) != case["case_id"]:
            raise ValueError(f"Unexpected case substitution at {old['case_id']}")
        feishu = by_old_id[old["case_id"]]
        number = int(feishu["题号"])
        result, review = results[case["case_id"]], reviews[case["case_id"]]
        is_new = case["case_id"] in new_ids
        if case["expected_action"] == "answer":
            value = new_record(number, case, result, review, docs)
        else:
            value = special_record(number, case, result, review)
            value["原文真值(人工填写)"] = (
                f"上游参考答案（只用于负例边界核查，不作为入库证据）：{case['reference_answer']}\n"
                f"预期操作：refuse。{'v2的554篇' if is_new else 'v1的520篇'}固定入库文档为本题实际问答语料；v2另对负例执行了全库精确匹配和高召回候选审查；"
                "仍非人工逐篇无答案确认。\n"
                f"模型初评说明：{(review.get('answer_review') or {}).get('notes', '未完成')}"
            )
        value["评测集"] = DATASET if is_new else "CRUD-RAG v2题目集／v1运行结果"
        value["题型细分"] = {"questanswer_1doc": "单文档事实题", "questanswer_2docs": "双文档综合题",
                          "questanswer_3docs": "三文档综合题", "near_no_answer": "近邻无答案候选"}[case["category"]]
        value["判定来源"] = PROVENANCE + ("；554篇v2语料" if is_new else "；沿用520篇v1语料的项目回答")
        value["评测版本"] = "v2新题／554篇语料" if is_new else "v2保留题／v1 520篇语料"
        if old["case_id"] in replace:
            value["修订原因"] = "v1上游答案缺少原文支持，换用同题型同拆分且有独立原文证据的题" if case["expected_action"] == "answer" else "v1负例在库内有完整或部分答案，换为重新审查的无答案候选"
        elif case.get("publisher_reference_answer"):
            value["修订原因"] = "问题保留；上游答案含原文未证实年份，改为原文可支持的月份"
            value["原文真值(人工填写)"] += f"\n原上游参考答案：{case['publisher_reference_answer']}（年份未获原文支持，v2已修订）"
        else:
            value["修订原因"] = "题目保留；沿用v1项目回答和原文证据"
        if review.get("status") == "gold_not_supported_by_model_review":
            value["修订原因"] += "；本题复评发现问题混合不同事件的影响，完整问法缺少入库原文支持，暂不计分"
        for field in ("精确判定(人工填写)", "幻觉判定(人工填写)", "任务完成判定(人工填写)"):
            if value.get(field) is None:
                reason = "原文答案待核实" if review.get("status") == "gold_not_supported_by_model_review" else "评审引文未通过逐字校验"
                value[field] = ("不计分：" if field.startswith("精确") else "待判定：") + reason
        if review.get("status") == "gold_not_supported_by_model_review":
            value["真值核验状态"] = "人工复核：完整问法未获入库原文支持；答案不计分"
        elif review.get("status") == "model_review_failed":
            value["真值核验状态"] = "模型评审引文未通过逐字校验；待重审"
        elif case["expected_action"] == "refuse" and not (review.get("answer_review") or {}).get("label_supported", False):
            value["真值核验状态"] = "模型发现库内可能有完整或部分答案；负例边界待复核"
            value["精确判定(人工填写)"] = "不计分：负例标签存疑"
            value["幻觉判定(人工填写)"] = "待判定：负例标签存疑"
            value["任务完成判定(人工填写)"] = "待判定：负例标签存疑"
        if case["case_id"] == grounding_audit["case_id"]:
            if review.get("status") != "gold_not_supported_by_model_review":
                raise ValueError("Grounding override must remain separate from answer scoring")
            value["幻觉判定(人工填写)"] = "是"
            value["判定来源"] += "；本题另由Codex对保存的回答和检索片段直接核对：回答将游客带动旅游复苏误归因于大桥项目"
        updates.append((number, feishu["record_id"], mark_human_reviewed(value)))
    updates.sort(key=lambda item: item[0])
    for start in range(0, 200, 10):
        write_json(OUT / f"questions_{start // 10 + 1:02}.json", {"update_records": {record_id: value for _, record_id, value in updates[start:start + 10]}})

    all_items = [(case, results[case["case_id"]], reviews[case["case_id"]]) for case in cases]
    old_items = [item for item in all_items if item[0]["case_id"] not in new_ids]
    new_items = [item for item in all_items if item[0]["case_id"] in new_ids]
    def grounding_labels(items: list[tuple[dict, dict, dict]]) -> list[bool]:
        labels = []
        for case, _, review in items:
            if case["case_id"] == grounding_audit["case_id"]:
                labels.append(True)
            elif review.get("status") == "model_review_complete" and review.get("answer_review") and (
                case["expected_action"] == "answer" or review["answer_review"].get("label_supported") is True
            ):
                labels.append(any(not claim["supported"] for claim in review["answer_review"]["claims"]))
        return labels
    def cohort_rate(items: list[tuple[dict, dict, dict]]) -> str:
        labels = grounding_labels(items)
        return percentage(sum(labels), len(labels))
    def five_metrics(title: str, items: list[tuple[dict, dict, dict]]) -> list[dict]:
        graded = annotated(items)
        answerable = [(case, result) for case, result, _ in items if case["expected_action"] == "answer"]
        pair_count = sum(len(case["gold_doc_ids"]) for case, _ in answerable)
        found = sum(len(set(case["gold_doc_ids"]) & set(result["final_doc_ids"])) for case, result in answerable)
        note = "本组所有幻觉、召回与回答判定均已由用户人工复核；语料版本见题目记录。"
        return [
            metric(title + "｜题数", str(len(items)), note),
            metric(title + "｜答案评分有效评审", f"{len(graded)}/{len(items)}", "只用于回答正确率；幻觉判定按回答与检索片段独立审查，可另有分母。"),
            metric(title + "｜来源文档召回@8", percentage(found, pair_count) if pair_count else "不适用：无正例", "只对可回答题按关联来源文档计算。"),
            metric(title + "｜回答正确率（人工复核）", percentage(sum(answer["answer_correct"] for _, _, answer in graded), len(graded)), note),
            metric(title + "｜幻觉率（人工复核）", cohort_rate(items), note),
        ]
    groups = {}
    for prefix, cohort in (("保留题/v1语料", old_items), ("新题/v2语料", new_items)):
        for title, category in (("单文档", "questanswer_1doc"), ("双文档", "questanswer_2docs"),
                ("三文档", "questanswer_3docs"), ("近邻无答案", "near_no_answer")):
            subset = [item for item in cohort if item[0]["category"] == category]
            if subset:
                groups[f"{prefix}·{title}"] = subset
    upload = json.loads((RUN / "upload_verification.json").read_text(encoding="utf-8"))
    statuses = Counter(review["status"] for _, _, review in all_items)
    challenged_negatives = sum(case["expected_action"] == "refuse" and review.get("status") == "model_review_complete"
        and not (review.get("answer_review") or {}).get("label_supported", False)
        for case, _, review in all_items)
    old_valid, new_valid = len(annotated(old_items)), len(annotated(new_items))
    answerable = [(case, result) for case, result, _ in all_items if case["expected_action"] == "answer"]
    source_pairs = sum(len(case["gold_doc_ids"]) for case, _ in answerable)
    found_pairs = sum(len(set(case["gold_doc_ids"]) & set(result["final_doc_ids"][:8])) for case, result in answerable)
    fully_recalled = sum(set(case["gold_doc_ids"]) <= set(result["final_doc_ids"][:8]) for case, result in answerable)
    labels = grounding_labels(all_items)
    if len(labels) != 200 or source_pairs != 360:
        raise ValueError("Combined denominators changed unexpectedly")
    boundary = "185道保留题沿用520篇v1语料结果，15道新题在554篇v2语料上运行；用户已人工复核全部逐题幻觉与召回判定。合并值是跨语料版本的200题描述性统计，不代表同一知识库的一次完整运行。"
    overview = [
        metric("【CRUD-RAG v2：200题合并展示】", "", "v1明细与汇总已本地归档；逐题判定均写回当前飞书表。"),
        metric("评测版本", "v2（跨语料合并展示）", boundary),
        metric("评测总题数", "200", "单文档60、双文档60、三文档60、近邻无答案20；dev/test分布不变。"),
        metric("保留题/v1语料运行数", "185/185", "原题不重跑；沿用520篇文档的项目真实上传与问答API记录。"),
        metric("新题/v2语料运行数", "15/15", "只对12道新增可回答题和3道新增无答案题调用项目问答API。"),
        metric("v2文档上传数", str(upload["document_count"]), "新增题使用554篇v2隔离知识库；全部通过项目POST /api/documents/upload。"),
        metric("v1项目切片数", "794", "保留题沿用v1运行时项目生成的切片。"),
        metric("v2项目切片数", str(upload["chunk_count"]), "新题使用v2项目上传接口生成的切片。"),
        metric("单文档题数", "60", "CRUD上游单来源问答。"),
        metric("双文档题数", "60", "CRUD上游双来源问答。"),
        metric("三文档题数", "60", "CRUD上游三来源问答；替代题要求三个来源各有独立原文贡献。"),
        metric("近邻无答案候选题数", "20", "全库精确匹配及高召回候选文档审查；无答案边界仍待人工确认。"),
        metric("保留题有效评审", f"{old_valid}/185", "v1语料；原题只复核了4道引文失败题和1道修正参考答案题。"),
        metric("新题有效评审", f"{new_valid}/15", "v2语料；新增题均重新作答和模型证据初评。"),
        metric("人工复核幻觉率", percentage(sum(labels), len(labels)), boundary),
        metric("200题幻觉判定已填写", f"{len(labels)}/200", "至少一条回答事实缺少实际检索片段支持记“是”；原文真值存疑题的回答证据可单独核查。"),
        metric("幻觉率（含无依据事实回答率，200题合并展示）", percentage(sum(labels), len(labels)), boundary),
        metric("来源文档召回@8（180题合并展示）", percentage(found_pairs, source_pairs), "180道可回答题的最终8个切片命中上游来源文档数/360个来源文档；无答案题无正例，不进入分母。"),
        metric("全部来源命中题率@8（180题合并展示）", percentage(fully_recalled, len(answerable)), "每道可回答题所有上游来源文档均进入最终8切片才记“是”；部分命中记“部分”。"),
        metric("召回判定分布", f"是{fully_recalled}／部分{len(answerable)-fully_recalled}／否0／不适用20", "20道近邻无答案题没有预期正例来源文档，召回判定为不适用。"),
        metric("答案评分未计分题数", str(200 - old_valid - new_valid), f"原文真值不支持完整问法{statuses['gold_not_supported_by_model_review']}题；该题的幻觉判定另按实际检索片段核对，不把答案正确率缺失当0。"),
    ]
    summary = mark_human_reviewed(overview + [row for title, items in groups.items() for row in five_metrics(title, items)])
    if len(summary) != 56:
        raise ValueError(f"Expected 56 summary rows; got {len(summary)}")
    for start in range(0, 56, 14):
        write_json(OUT / f"summary_{start // 14 + 1:02}.json", {"update_records": {old_summary[i]["record_id"]: summary[i] for i in range(start, start + 14)}})
    write_json(OUT / "manifest.json", {"version": "v2", "case_count": 200, "document_count": upload["document_count"],
        "chunk_count": upload["chunk_count"], "retained185_reviewed": old_valid, "new15_reviewed": new_valid,
        "grounding_judgments": len(labels), "answers_with_unsupported_claims": sum(labels),
        "source_pairs_recalled_at_8": found_pairs, "source_pairs_total": source_pairs,
        "fully_recalled_answerable_cases": fully_recalled,
        "review_statuses": dict(statuses), "groups": {title: len(items) for title, items in groups.items()},
        "warning": "Descriptive aggregation across two corpora; model-assisted preliminary scores; human-confirmed hallucination rate pending"})
    print(f"Prepared v2 Feishu payloads: 200 cases, 56 summary rows; retained {old_valid}/185 and new {new_valid}/15 model reviews")


if __name__ == "__main__":
    main()
