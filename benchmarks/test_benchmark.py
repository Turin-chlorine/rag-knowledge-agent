"""Small checks for benchmark leakage and fail-closed scoring."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluate import merge_gold_reviews, score_reviews, summarize_retrieval
from prepare import crud_rows, source_evidence_candidate
from adjudicate import validate_answer, validate_gold


class BenchmarkTests(unittest.TestCase):
    def test_crud_selection_keeps_source_events_distinct(self):
        def item(number):
            return {"ID": str(number), "questions": f"问题{number}", "answers": "回答", "news1": "一" * 100, "news2": "二" * 100, "news3": "三" * 100}
        raw = {
            "questanswer_1doc": [item(1), item(2), item(3)],
            "questanswer_2docs": [item(1), item(2), item(4)],
            "questanswer_3docs": [item(2), item(4), item(5)],
        }
        selected = crud_rows(raw, per_type=1, seed=42)
        self.assertEqual(3, len(selected))
        self.assertEqual(3, len({row["ID"] for _, row in selected}))

    def test_candidate_quote_is_exact_source_span_and_unreviewed(self):
        text = "甲公司发布新产品。乙公司并未发布新产品。"
        candidate = source_evidence_candidate("谁发布新产品？", "甲公司", text)
        self.assertEqual(candidate["quote"], text[candidate["start"]:candidate["end"]].strip())
        self.assertEqual("needs_human_review", candidate["review_status"])

    def test_unreviewed_answer_never_becomes_zero_hallucination(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            results = folder / "results.jsonl"
            reviews = folder / "reviews.jsonl"
            row = {"case_id": "c1", "category": "single", "expected_action": "answer", "question": "Q", "answer": "A"}
            results.write_text(json.dumps(row) + "\n", encoding="utf-8")
            reviews.write_text(json.dumps({"case_id": "c1", "gold_evidence_verified": None, "answer_correct": True, "action_correct": True, "citation_correct": True, "supported_claims": 1, "unsupported_claims": 0}) + "\n", encoding="utf-8")
            report_path = score_reviews(results, reviews)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(0, report["reviewed"])
            self.assertIsNone(report["overall"]["unsupported_claim_rate"])
            self.assertIn("待人工核实", (folder / "results_scored_report.md").read_text(encoding="utf-8"))

    def test_retrieval_summary_separates_unverified_negative_cases(self):
        results = [
            {"category": "single", "gold_doc_ids": ["a"], "coarse_recall_at_20": 1.0, "final_recall_at_8": 0.0, "rerank_applied": True},
            {"category": "near_no_answer", "gold_doc_ids": [], "coarse_recall_at_20": None, "final_recall_at_8": None, "rerank_applied": True},
        ]
        report = summarize_retrieval(results)
        self.assertEqual(1, report["overall"]["source_linked_cases"])
        self.assertEqual(1.0, report["overall"]["coarse_source_recall_at_20"])
        self.assertIsNone(report["by_category"]["near_no_answer"]["final_source_recall_at_8"])

    def test_scoring_requires_verbatim_source_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "crud"
            run_dir = root / "runs"
            docs_dir = root / "prepared" / "documents"
            docs_dir.mkdir(parents=True)
            run_dir.mkdir()
            (docs_dir / "d.txt").write_text("原文明确指出甲公司发布了产品。", encoding="utf-8")
            (docs_dir.parent / "documents.jsonl").write_text(json.dumps({"doc_id": "d", "filename": "d.txt"}) + "\n", encoding="utf-8")
            result_path = run_dir / "results.jsonl"
            result = {"case_id": "c1", "track": "crud", "category": "single", "expected_action": "answer", "gold_doc_ids": ["d"]}
            result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
            review_path = run_dir / "reviews.jsonl"
            review = {"case_id": "c1", "gold_evidence_verified": True, "verified_evidence": [{"doc_id": "d", "quote": "甲公司发布了产品"}], "answer_correct": True, "action_correct": True, "citation_correct": True, "supported_claims": 1, "unsupported_claims": 0, "reviewer": "human"}
            review_path.write_text(json.dumps(review, ensure_ascii=False) + "\n", encoding="utf-8")
            report = json.loads(score_reviews(result_path, review_path).read_text(encoding="utf-8"))
            self.assertEqual(1, report["reviewed"])
            self.assertEqual(0.0, report["overall"]["unsupported_claim_rate"])
            review["verified_evidence"][0]["quote"] = "不存在的引文"
            review_path.write_text(json.dumps(review, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                score_reviews(result_path, review_path)

    def test_gold_reviews_merge_without_pretending_answers_are_graded(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            results = folder / "run.jsonl"
            gold = folder / "gold.jsonl"
            results.write_text(json.dumps({"case_id": "c1", "evidence_candidates": [{"quote": "事实"}]}, ensure_ascii=False) + "\n", encoding="utf-8")
            gold.write_text(json.dumps({"case_id": "c1", "gold_evidence_verified": True, "verified_evidence": [{"doc_id": "d", "quote": "事实"}], "reviewer": "human"}, ensure_ascii=False) + "\n", encoding="utf-8")
            merged_path = merge_gold_reviews(results, gold)
            merged = json.loads(merged_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(merged["gold_evidence_verified"])
            self.assertIsNone(merged["answer_correct"])

    def test_model_review_rejects_invented_gold_and_answer_quotes(self):
        gold = {"gold_supported": True, "verified_answer": "甲发布了产品", "evidence": [{"doc_id": "d", "quote": "甲发布了产品"}]}
        validate_gold(gold, {"d": "原文：甲发布了产品。"})
        gold["evidence"][0]["quote"] = "乙发布了产品"
        with self.assertRaises(ValueError):
            validate_gold(gold, {"d": "原文：甲发布了产品。"})
        answer = {"task_completion": "是", "answer_correct": True, "action_correct": True, "citation_correct": True, "relevant_source_indices": [1], "claims": [{"text": "甲发布了产品", "supported": True, "source_index": 1, "quote": "甲发布了产品"}]}
        validate_answer(answer, [{"index": 1, "content": "原文：甲发布了产品。"}])
        answer["claims"][0]["quote"] = "乙发布了产品"
        with self.assertRaises(ValueError):
            validate_answer(answer, [{"index": 1, "content": "原文：甲发布了产品。"}])


if __name__ == "__main__":
    unittest.main()
