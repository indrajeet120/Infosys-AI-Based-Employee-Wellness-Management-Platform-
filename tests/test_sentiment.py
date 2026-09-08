"""
Unit tests for the VADER Sentiment Analysis module.
"""

import pytest
from services.sentiment import VaderSentimentAnalyzer, analyze_sentiment


@pytest.fixture
def analyzer():
    return VaderSentimentAnalyzer()


class TestVaderSentiment:
    def test_positive_sentences(self, analyzer):
        pos_sentences = [
            "I absolutely love this product.",
            "This is an amazing experience.",
            "The service was fantastic.",
        ]
        for s in pos_sentences:
            res = analyzer.analyze(s)
            assert res["sentiment"] == "Positive", f"Failed for '{s}': {res}"
            assert res["compound"] >= 0.05
            assert res["pos"] > 0.0
            assert "neg" in res and "neu" in res and "pos" in res and "compound" in res

    def test_negative_sentences(self, analyzer):
        neg_sentences = [
            "I hate this product.",
            "The service was terrible.",
            "I am extremely disappointed.",
        ]
        for s in neg_sentences:
            res = analyzer.analyze(s)
            assert res["sentiment"] == "Negative", f"Failed for '{s}': {res}"
            assert res["compound"] <= -0.05
            assert res["neg"] > 0.0

    def test_neutral_sentences(self, analyzer):
        neu_sentences = [
            "The meeting is scheduled for Monday.",
            "The product was delivered today.",
            "The system has three modules.",
        ]
        for s in neu_sentences:
            res = analyzer.analyze(s)
            assert res["sentiment"] == "Neutral", f"Failed for '{s}': {res}"
            assert -0.05 < res["compound"] < 0.05

    def test_scores_are_dynamic_and_not_hardcoded(self, analyzer):
        res1 = analyzer.analyze("I love this.")
        res2 = analyzer.analyze("I hate this.")
        res3 = analyzer.analyze("This is standard.")

        # Ensure different texts produce different compound scores
        assert res1["compound"] != res2["compound"]
        assert res1["compound"] != res3["compound"]
        assert res2["compound"] != res3["compound"]

    def test_threshold_boundaries(self, analyzer):
        assert analyzer.classify_compound(0.05) == "Positive"
        assert analyzer.classify_compound(0.06) == "Positive"
        assert analyzer.classify_compound(-0.05) == "Negative"
        assert analyzer.classify_compound(-0.06) == "Negative"
        assert analyzer.classify_compound(0.0) == "Neutral"
        assert analyzer.classify_compound(0.049) == "Neutral"
        assert analyzer.classify_compound(-0.049) == "Neutral"

    def test_empty_input_handling(self, analyzer):
        res = analyzer.analyze("")
        assert res["sentiment"] == "Neutral"
        assert res["compound"] == 0.0
