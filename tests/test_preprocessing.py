"""
Unit tests for the Text Preprocessing module.
"""

import pytest
from services.preprocessing import TextPreprocessor, preprocess_text


@pytest.fixture
def preprocessor():
    return TextPreprocessor(preserve_negations=True)


class TestTextPreprocessing:
    def test_normal_text(self, preprocessor):
        raw = "I really love this product!"
        processed = preprocessor.preprocess(raw)
        assert "love" in processed
        assert "product" in processed
        assert "!" not in processed

    def test_repeated_spaces(self, preprocessor):
        raw = "I   really    love    this."
        processed = preprocessor.preprocess(raw)
        assert "   " not in processed
        assert processed.strip() == "really love"

    def test_punctuation_handling(self, preprocessor):
        raw = "This product is excellent!!!"
        processed = preprocessor.preprocess(raw)
        assert "!" not in processed
        assert "product" in processed
        assert "excellent" in processed

    def test_special_characters_and_hashtags(self, preprocessor):
        raw = "Excellent @product #happy"
        processed = preprocessor.preprocess(raw)
        assert "@" not in processed
        assert "#" not in processed
        assert "product" in processed
        assert "happy" in processed
        assert "excellent" in processed

    def test_empty_and_whitespace_input(self, preprocessor):
        assert preprocessor.preprocess("") == ""
        assert preprocessor.preprocess("    ") == ""
        assert preprocessor.preprocess(None) == ""

    def test_different_casing(self, preprocessor):
        raw = "GOOD Good good"
        processed = preprocessor.preprocess(raw)
        tokens = processed.split()
        assert all(t == "good" for t in tokens)

    def test_negation_preservation(self, preprocessor):
        raw = "I am not happy."
        processed = preprocessor.preprocess(raw)
        assert "not" in processed
        assert "happy" in processed
        # Ensure 'not' was NOT stripped out as a stopword
        assert processed == "not happy"

    def test_various_negation_words(self, preprocessor):
        sentences = [
            ("We never received the delivery", "never"),
            ("There is no solution yet", "no"),
            ("It cannot work properly", "not"),
        ]
        for sentence, expected_neg in sentences:
            processed = preprocessor.preprocess(sentence)
            assert expected_neg in processed

    def test_short_and_long_text(self, preprocessor):
        # Short
        short_text = "Good"
        assert preprocessor.preprocess(short_text) == "good"

        # Long
        long_text = """
        Natural language processing provides automated text analysis.
        The system continuously evaluates customer feedback, product reviews, and social sentiments.
        It generates reliable scores and high quality analytics for data science teams.
        """
        processed_long = preprocessor.preprocess(long_text)
        assert len(processed_long) > 0
        assert "nlp" not in processed_long or "system" in processed_long

    def test_html_and_url_cleaning(self, preprocessor):
        raw = "<p>Visit our website at https://example.com for great deals!</p>"
        processed = preprocessor.preprocess(raw)
        assert "<p>" not in processed
        assert "https" not in processed
        assert "deal" in processed or "great" in processed
