"""
Sentiment analysis service using NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner).
Provides dynamic sentiment scoring and threshold-based classification.
"""

from typing import Dict, Union
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from services.preprocessing import ensure_nltk_resources


class VaderSentimentAnalyzer:
    """
    Wrapper around NLTK SentimentIntensityAnalyzer for baseline sentiment classification.
    """

    def __init__(self, pos_threshold: float = 0.05, neg_threshold: float = -0.05):
        ensure_nltk_resources()
        self.analyzer = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def classify_compound(self, compound_score: float) -> str:
        """
        Classifies compound score according to standard VADER rules:
        - compound >= 0.05  -> Positive
        - compound <= -0.05 -> Negative
        - otherwise         -> Neutral
        """
        if compound_score >= self.pos_threshold:
            return "Positive"
        elif compound_score <= self.neg_threshold:
            return "Negative"
        else:
            return "Neutral"

    def analyze(self, text: str) -> Dict[str, Union[float, str]]:
        """
        Computes polarity scores and classification for the provided text.

        Returns:
            Dictionary containing:
            - neg: float (0.0 to 1.0)
            - neu: float (0.0 to 1.0)
            - pos: float (0.0 to 1.0)
            - compound: float (-1.0 to 1.0)
            - sentiment: str ('Positive', 'Negative', 'Neutral')
        """
        if not text or not str(text).strip():
            # For empty or invalid text that reaches sentiment analyzer
            return {
                "neg": 0.0,
                "neu": 0.0,
                "pos": 0.0,
                "compound": 0.0,
                "sentiment": "Neutral",
            }

        scores = self.analyzer.polarity_scores(str(text))
        compound = round(scores["compound"], 4)
        sentiment = self.classify_compound(compound)

        return {
            "neg": round(scores["neg"], 4),
            "neu": round(scores["neu"], 4),
            "pos": round(scores["pos"], 4),
            "compound": compound,
            "sentiment": sentiment,
        }


# Singleton analyzer instance for convenience
_default_analyzer: Union[VaderSentimentAnalyzer, None] = None


def analyze_sentiment(text: str) -> Dict[str, Union[float, str]]:
    """
    Convenience function for analyzing sentiment of a text string.
    """
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = VaderSentimentAnalyzer()
    return _default_analyzer.analyze(text)
