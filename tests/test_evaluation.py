"""
Unit tests for Model Evaluation and Model Comparison services.
"""

import pytest
from services.evaluation import evaluate_emotion_model, compare_models
from services.config import TEST_DATA_PATH


class TestModelEvaluation:
    def test_evaluate_distilbert_metrics(self):
        results = evaluate_emotion_model("distilbert", test_data_path=TEST_DATA_PATH)
        assert results["model"] == "DISTILBERT"
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "macro_f1" in results
        assert "emotion_metrics" in results
        assert results["macro_f1"] > 0.0

    def test_evaluate_bert_metrics(self):
        results = evaluate_emotion_model("bert", test_data_path=TEST_DATA_PATH)
        assert results["model"] == "BERT"
        assert results["accuracy"] > 0.0
        assert results["macro_f1"] > 0.0

    def test_compare_models_generates_valid_report(self):
        comp = compare_models(test_data_path=TEST_DATA_PATH)
        assert "better_performing_model" in comp
        assert comp["better_performing_model"] in ["BERT", "DistilBERT"]
        assert "metrics_summary" in comp
        assert "Accuracy" in comp["metrics_summary"]
        assert "Macro F1" in comp["metrics_summary"]
