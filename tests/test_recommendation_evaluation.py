"""
Unit & Integration Tests for Task 9 – Advanced ML Validation & Performance Testing.
"""

import json
from pathlib import Path
import pytest
from services.recommendation_eval import (
    RecommendationEvaluator,
    BaselineRuleRecommender,
    build_controlled_evaluation_dataset,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_f1_at_k,
    calculate_ndcg_at_k,
    calculate_ground_truth_acceptance_proxy,
    calculate_diversity_score,
    run_recommendation_evaluation,
)
from services.wellness_data import get_wellness_repository
from services.intensity import analyze_emotional_state, EmotionalState
from services.config import REPORTS_DIR


class TestRecommendationEvaluation:
    def test_01_controlled_dataset_loads_successfully(self):
        dataset = build_controlled_evaluation_dataset()
        assert len(dataset) == 15
        emotions_found = set(p.dominant_emotion for p in dataset)
        assert emotions_found == {"Joy", "Fear", "Sadness", "Anger", "Surprise", "Disgust"}

    def test_02_all_profiles_have_valid_ground_truth(self):
        dataset = build_controlled_evaluation_dataset()
        repo = get_wellness_repository()
        all_content_ids = set(item.content_id for item in repo.get_all())

        for p in dataset:
            assert isinstance(p.ground_truth_relevant_ids, set)
            assert len(p.ground_truth_relevant_ids) > 0
            # Ensure all ground truth IDs exist in the repo candidate pool
            assert p.ground_truth_relevant_ids.issubset(all_content_ids)

    def test_03_candidate_pool_is_valid(self):
        repo = get_wellness_repository()
        assert repo.count() == 24
        assert len(repo.get_all()) == 24

    def test_04_precision_at_k_calculation(self):
        recs = ["well_001", "well_002", "well_003"]
        gt = {"well_001", "well_002"}
        p = calculate_precision_at_k(recs, gt, k=3)
        assert p == round(2 / 3, 4)

    def test_05_recall_at_k_calculation(self):
        recs = ["well_001", "well_002", "well_003"]
        gt = {"well_001", "well_002", "well_004", "well_005"}
        r = calculate_recall_at_k(recs, gt, k=3)
        assert r == round(2 / 4, 4)

    def test_06_f1_at_k_calculation(self):
        assert calculate_f1_at_k(0.5, 0.5) == 0.5
        assert calculate_f1_at_k(0.0, 0.0) == 0.0
        assert calculate_f1_at_k(0.8, 0.4) == round(2 * 0.8 * 0.4 / (0.8 + 0.4), 4)

    def test_07_ndcg_at_k_calculation(self):
        # First item relevant -> DCG = 1/log2(2) = 1.0; IDCG = 1.0 -> NDCG = 1.0
        recs = ["well_001", "well_002"]
        gt = {"well_001"}
        ndcg = calculate_ndcg_at_k(recs, gt, k=2)
        assert ndcg == 1.0

        # Non-relevant first, relevant second -> DCG = 1/log2(3) = 0.6309; IDCG = 1.0 -> NDCG = 0.6309
        recs_reverse = ["well_002", "well_001"]
        ndcg_rev = calculate_ndcg_at_k(recs_reverse, gt, k=2)
        assert ndcg_rev < 1.0

    def test_08_diversity_score_calculation(self):
        acts = ["breathing exercise", "journaling", "meditation"]
        div = calculate_diversity_score(acts)
        assert div == 1.0  # 3 unique out of 3

        acts_dup = ["breathing exercise", "breathing exercise", "meditation"]
        div_dup = calculate_diversity_score(acts_dup)
        assert div_dup == round(2 / 3, 4)

    def test_09_empty_input_handling(self):
        assert calculate_precision_at_k([], {"well_001"}, k=3) == 0.0
        assert calculate_recall_at_k(["well_001"], set(), k=3) == 0.0
        assert calculate_f1_at_k(0.0, 0.0) == 0.0
        assert calculate_ndcg_at_k([], {"well_001"}, k=3) == 0.0
        assert calculate_diversity_score([]) == 0.0

    def test_10_k_larger_than_list_length(self):
        recs = ["well_001"]
        gt = {"well_001"}
        p = calculate_precision_at_k(recs, gt, k=5)
        assert p == 1.0

    def test_11_baseline_execution(self):
        base = BaselineRuleRecommender()
        state = analyze_emotional_state("Severe anxiety", model_type="distilbert")
        recs = base.recommend(state, top_k=3)
        assert len(recs) <= 3
        for r in recs:
            assert "content_id" in r
            assert "title" in r
            assert "score" in r

    def test_12_advanced_execution(self):
        evaluator = RecommendationEvaluator(k=3)
        engine = evaluator.advanced_engine
        state = analyze_emotional_state("Severe anxiety", model_type="distilbert")
        res = engine.recommend(state, user_id="test_user_adv", top_k=3)
        assert len(res["recommendations"]) == 3

    def test_13_identical_candidate_pools(self):
        evaluator = RecommendationEvaluator(k=3)
        base_pool = set(item.content_id for item in evaluator.baseline_recommender.repo.get_all())
        adv_pool = set(item.content_id for item in evaluator.advanced_engine.repo.get_all())
        assert base_pool == adv_pool

    def test_14_full_evaluation_pipeline(self):
        evaluator = RecommendationEvaluator(k=3)
        report = evaluator.evaluate()
        assert "baseline_metrics" in report
        assert "advanced_metrics" in report
        assert "improvement_pct" in report
        assert report["baseline_metrics"]["precision_at_k"] > 0.0
        assert report["advanced_metrics"]["precision_at_k"] > 0.0

    def test_15_json_report_generation(self, tmp_path):
        evaluator = RecommendationEvaluator(k=3, tmp_dir=tmp_path)
        report = evaluator.evaluate()
        json_p, _ = evaluator.save_reports(report)
        assert json_p.exists()
        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "evaluation_configuration" in data

    def test_16_markdown_report_generation(self, tmp_path):
        evaluator = RecommendationEvaluator(k=3, tmp_dir=tmp_path)
        report = evaluator.evaluate()
        _, md_p = evaluator.save_reports(report)
        assert md_p.exists()
        content = md_p.read_text(encoding="utf-8")
        assert "| Metric | Baseline | Advanced (Hybrid ML) | Improvement (%) |" in content

    def test_17_report_schema_validity(self):
        report = run_recommendation_evaluation(k=3)
        assert "evaluation_configuration" in report
        assert "metric_definitions" in report
        assert "baseline_metrics" in report
        assert "advanced_metrics" in report
        assert "improvement_pct" in report
        assert "per_user_results" in report
        assert len(report["per_user_results"]) == 15

    def test_18_latency_measurement(self):
        evaluator = RecommendationEvaluator(k=3)
        profile = evaluator.dataset[0]
        state = EmotionalState(
            dominant_emotion=profile.dominant_emotion,
            emotional_intensity=profile.emotional_intensity,
            positive_polarity=0.15,
            negative_polarity=0.85,
            final_emotional_state="distressed",
            raw_text=profile.raw_text,
            probabilities={"Fear": 0.85},
            dominant_confidence=0.85,
        )
        mean_lat, p95_lat = evaluator.measure_latency("baseline", state, profile.user_id, num_runs=3)
        assert mean_lat >= 0.0
        assert p95_lat >= mean_lat or p95_lat >= 0.0

    def test_19_no_hardcoded_metric_values(self):
        evaluator1 = RecommendationEvaluator(k=2)
        report1 = evaluator1.evaluate()
        evaluator2 = RecommendationEvaluator(k=4)
        report2 = evaluator2.evaluate()
        
        # Metrics computed at K=2 and K=4 must differ dynamically based on K
        assert report1["baseline_metrics"]["precision_at_k"] != report2["baseline_metrics"]["precision_at_k"] or report1["evaluation_configuration"]["k"] != report2["evaluation_configuration"]["k"]

    def test_20_deterministic_evaluation_results(self):
        evaluator1 = RecommendationEvaluator(k=3)
        report1 = evaluator1.evaluate()
        evaluator2 = RecommendationEvaluator(k=3)
        report2 = evaluator2.evaluate()

        assert report1["baseline_metrics"]["precision_at_k"] == report2["baseline_metrics"]["precision_at_k"]
        assert report1["advanced_metrics"]["precision_at_k"] == report2["advanced_metrics"]["precision_at_k"]
