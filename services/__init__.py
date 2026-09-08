"""
Services package for Text Sentiment Analysis and Deep Emotion Classification.
"""
from services.ingestion import ingest_text, ingest_txt_file, ingest_csv_file, IngestedItem
from services.preprocessing import preprocess_text, TextPreprocessor
from services.sentiment import analyze_sentiment, VaderSentimentAnalyzer
from services.emotion import analyze_emotion, get_emotion_classifier, EmotionClassifier, EmotionPrediction
from services.evaluation import evaluate_emotion_model, compare_models
from services.isear_validation import validate_on_isear_benchmark
from services.reporting import (
    generate_sentiment_report,
    generate_complete_report,
    calculate_summary_stats,
    process_pipeline_items,
)

__all__ = [
    "ingest_text",
    "ingest_txt_file",
    "ingest_csv_file",
    "IngestedItem",
    "preprocess_text",
    "TextPreprocessor",
    "analyze_sentiment",
    "VaderSentimentAnalyzer",
    "analyze_emotion",
    "get_emotion_classifier",
    "EmotionClassifier",
    "EmotionPrediction",
    "evaluate_emotion_model",
    "compare_models",
    "validate_on_isear_benchmark",
    "generate_sentiment_report",
    "generate_complete_report",
    "calculate_summary_stats",
    "process_pipeline_items",
]
