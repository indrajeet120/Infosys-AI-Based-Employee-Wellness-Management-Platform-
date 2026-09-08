"""
Unit tests for the Deep Emotion Classification inference service (BERT & DistilBERT).
"""

import pytest
from services.config import EMOTIONS, EMOTION_DISPLAY_NAMES
from services.emotion import EmotionClassifier, get_emotion_classifier


@pytest.fixture(scope="module")
def distilbert_classifier():
    return get_emotion_classifier("distilbert")


@pytest.fixture(scope="module")
def bert_classifier():
    return get_emotion_classifier("bert")


class TestEmotionClassification:
    def test_model_and_tokenizer_loaded(self, distilbert_classifier, bert_classifier):
        assert distilbert_classifier.is_loaded is True
        assert distilbert_classifier.model is not None
        assert distilbert_classifier.tokenizer is not None

        assert bert_classifier.is_loaded is True
        assert bert_classifier.model is not None
        assert bert_classifier.tokenizer is not None

    def test_single_emotion_joy_prediction(self, distilbert_classifier):
        text = "I received my dream job offer today and cannot stop smiling!"
        pred = distilbert_classifier.predict(text)
        assert pred.is_valid is True
        assert pred.primary_emotion == "Joy"
        assert pred.probabilities["Joy"] > 0.4
        assert len(pred.probabilities) == 6

    def test_single_emotion_anger_prediction(self, distilbert_classifier):
        text = "I am furious that my order was cancelled without any explanation!"
        pred = distilbert_classifier.predict(text)
        assert pred.is_valid is True
        assert pred.primary_emotion == "Anger"
        assert pred.probabilities["Anger"] > 0.4

    def test_multi_emotion_prediction(self, distilbert_classifier):
        text = "I am happy about winning the race but scared about defending the title."
        pred = distilbert_classifier.predict(text, threshold=0.30)
        assert pred.is_valid is True
        assert len(pred.probabilities) == 6
        detected_names = [d["emotion"] for d in pred.detected_emotions]
        # Multi-label should capture both joy and fear signals
        assert "Joy" in detected_names or "Fear" in detected_names

    def test_all_six_emotions_present_in_probabilities(self, distilbert_classifier):
        text = "Test sample sentence."
        pred = distilbert_classifier.predict(text)
        for emo in EMOTIONS:
            disp_name = EMOTION_DISPLAY_NAMES[emo]
            assert disp_name in pred.probabilities
            p = pred.probabilities[disp_name]
            assert 0.0 <= p <= 1.0

    def test_dynamic_confidence_scores_not_hardcoded(self, distilbert_classifier):
        pred1 = distilbert_classifier.predict("I feel so joyful and ecstatic!")
        pred2 = distilbert_classifier.predict("This rotten food smells disgusting.")
        assert pred1.probabilities["Joy"] != pred2.probabilities["Joy"]
        assert pred1.primary_emotion != pred2.primary_emotion

    def test_configurable_threshold_filtering(self, distilbert_classifier):
        text = "I am terrified and disgusted by what I saw."
        pred_low_thresh = distilbert_classifier.predict(text, threshold=0.10)
        pred_high_thresh = distilbert_classifier.predict(text, threshold=0.90)
        assert len(pred_low_thresh.detected_emotions) >= len(pred_high_thresh.detected_emotions)

    def test_edge_case_empty_and_whitespace(self, distilbert_classifier):
        pred_empty = distilbert_classifier.predict("")
        assert pred_empty.is_valid is False
        assert "empty" in pred_empty.error_message.lower()

        pred_ws = distilbert_classifier.predict("    \n\t  ")
        assert pred_ws.is_valid is False

        pred_none = distilbert_classifier.predict(None)
        assert pred_none.is_valid is False

    def test_edge_case_emojis_and_informal_text(self, distilbert_classifier):
        text = "OMG this is soooo good!!! I love it 😊🎉"
        pred = distilbert_classifier.predict(text)
        assert pred.is_valid is True
        assert pred.primary_emotion == "Joy"

    def test_edge_case_long_paragraph(self, distilbert_classifier):
        long_text = """
        The unexpected announcement came early in the morning when the team was reviewing the final quarterly results.
        Everyone was stunned by the massive achievements and celebrated with immense happiness throughout the company.
        """ * 3
        pred = distilbert_classifier.predict(long_text)
        assert pred.is_valid is True
        assert pred.primary_confidence > 0.0
