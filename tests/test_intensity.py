"""
Unit tests for Task 1: Emotion Intensity and Emotional State Analysis.
"""

import pytest
from services.intensity import (
    analyze_emotional_state,
    calculate_emotional_intensity,
    calculate_polarity,
    calculate_distribution_entropy,
    detect_mixed_emotions,
    determine_severity_level,
    EmotionalState,
)


class TestEmotionIntensity:
    def test_dominant_emotion_and_confidence_range(self):
        text = "I received my dream job offer today and cannot stop smiling!"
        state = analyze_emotional_state(text, model_type="distilbert")
        assert state.is_valid is True
        assert state.dominant_emotion == "Joy"
        assert 0.0 <= state.dominant_confidence <= 1.0

    def test_top_k_emotions_structure(self):
        text = "I am extremely furious and disgusted by the terrible customer service!"
        state = analyze_emotional_state(text, model_type="distilbert", top_k=3)
        assert len(state.top_emotions) == 3
        for item in state.top_emotions:
            assert "emotion" in item
            assert "confidence" in item
            assert 0.0 <= item["confidence"] <= 1.0
        # Confidences should be in descending order
        confidences = [e["confidence"] for e in state.top_emotions]
        assert confidences == sorted(confidences, reverse=True)

    def test_emotional_intensity_normalized_range(self):
        text = "I am deeply sad and feeling hopeless."
        state = analyze_emotional_state(text, model_type="distilbert")
        assert 0.0 <= state.emotional_intensity <= 1.0

    def test_positive_and_negative_polarity(self):
        probs_joy = {"Joy": 0.85, "Sadness": 0.10, "Anger": 0.05, "Fear": 0.05, "Surprise": 0.20, "Disgust": 0.05}
        pol_joy = calculate_polarity(probs_joy)
        assert pol_joy["positive_polarity"] > pol_joy["negative_polarity"]

        probs_anger = {"Joy": 0.05, "Sadness": 0.40, "Anger": 0.88, "Fear": 0.50, "Surprise": 0.10, "Disgust": 0.70}
        pol_anger = calculate_polarity(probs_anger)
        assert pol_anger["negative_polarity"] > pol_anger["positive_polarity"]

    def test_mixed_emotion_detection(self):
        # Mixed: high joy and fear simultaneously
        probs_mixed = {"Joy": 0.70, "Sadness": 0.10, "Anger": 0.05, "Fear": 0.65, "Surprise": 0.20, "Disgust": 0.05}
        is_mixed = detect_mixed_emotions(probs_mixed, positive_polarity=0.70, negative_polarity=0.65, threshold=0.30)
        assert is_mixed is True

        # Unmixed: purely joy
        probs_unmixed = {"Joy": 0.90, "Sadness": 0.05, "Anger": 0.02, "Fear": 0.02, "Surprise": 0.10, "Disgust": 0.01}
        is_unmixed = detect_mixed_emotions(probs_unmixed, positive_polarity=0.90, negative_polarity=0.05, threshold=0.30)
        assert is_unmixed is False

    def test_severity_levels(self):
        assert determine_severity_level(0.20, "Joy") == "Low"
        assert determine_severity_level(0.50, "Sadness") == "Moderate"
        assert determine_severity_level(0.75, "Anger") == "High"
        assert determine_severity_level(0.95, "Fear") == "Very High"

    def test_empty_and_whitespace_input_handling(self):
        state_empty = analyze_emotional_state("")
        assert state_empty.is_valid is False
        assert "empty" in state_empty.error_message.lower()

        state_ws = analyze_emotional_state("    \n\t  ")
        assert state_ws.is_valid is False

        state_none = analyze_emotional_state(None)
        assert state_none.is_valid is False

    def test_emotional_state_serializable_to_dict(self):
        text = "Everything went wrong today and I am panicking."
        state = analyze_emotional_state(text, model_type="distilbert")
        state_dict = state.to_dict()
        assert isinstance(state_dict, dict)
        assert "dominant_emotion" in state_dict
        assert "emotional_intensity" in state_dict
        assert "final_emotional_state" in state_dict
