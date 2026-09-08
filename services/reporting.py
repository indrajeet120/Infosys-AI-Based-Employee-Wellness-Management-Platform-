"""
Reporting service for Text Sentiment & Deep Emotion Analysis.
Generates comprehensive tabular reports and aggregated summary statistics combining
Milestone 1 (VADER Sentiment) and Milestone 2 (Transformer Multi-label Emotion Analysis).
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from services.config import EMOTIONS, EMOTION_DISPLAY_NAMES, DEFAULT_THRESHOLD
from services.emotion import analyze_emotion, get_emotion_classifier
from services.ingestion import IngestedItem
from services.preprocessing import preprocess_text
from services.sentiment import analyze_sentiment


def process_pipeline_items(
    items: List[IngestedItem],
    analyze_source_text: str = "original",
    include_emotion: bool = False,
    emotion_model_type: str = "bert",
    emotion_threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the end-to-end sentiment (and optional emotion) pipeline:
    Ingestion validation -> Preprocessing -> VADER Sentiment Scoring -> Transformer Emotion -> Reporting.

    Args:
        items: List of IngestedItem instances.
        analyze_source_text: Whether VADER should score 'original' or 'processed' text.
                             Default is 'original'. Transformers always use the original input text.
        include_emotion: Whether to also run BERT / DistilBERT multi-label emotion classification.
        emotion_model_type: 'bert' or 'distilbert'.
        emotion_threshold: Multi-label probability cutoff (default 0.50).

    Returns:
        (report_df, summary_stats_dict)
    """
    records = []
    total_samples = len(items)
    valid_samples = 0
    invalid_samples = 0
    analyzed_samples = 0
    pos_count = 0
    neg_count = 0
    neu_count = 0
    emotion_counts = {EMOTION_DISPLAY_NAMES[e]: 0 for e in EMOTIONS}

    classifier = None
    if include_emotion:
        try:
            classifier = get_emotion_classifier(emotion_model_type)
        except Exception:
            classifier = None

    for item in items:
        if not item.is_valid:
            invalid_samples += 1
            row_dict = {
                "ID": item.id,
                "Input Text": item.text,
                "Processed Text": "",
                "Sentiment": "Invalid",
                "Positive": None,
                "Negative": None,
                "Neutral": None,
                "Compound": None,
                "Status": "Invalid",
                "Error": item.error_message or "Validation failed",
            }
            if include_emotion:
                row_dict.update({
                    "Emotion Model": emotion_model_type.upper(),
                    "Primary Emotion": "N/A",
                    "Primary Confidence": None,
                    "Detected Emotions": "N/A",
                    "Combined Analysis": "Invalid Input",
                })
                for emo in EMOTIONS:
                    row_dict[f"Prob_{EMOTION_DISPLAY_NAMES[emo]}"] = None
            records.append(row_dict)
            continue

        valid_samples += 1
        # Preprocess text (Milestone 1)
        processed = preprocess_text(item.text)

        # Sentiment Analysis (Milestone 1)
        text_to_score = item.text if analyze_source_text == "original" else processed
        scores = analyze_sentiment(text_to_score)

        analyzed_samples += 1
        sentiment = scores["sentiment"]
        if sentiment == "Positive":
            pos_count += 1
        elif sentiment == "Negative":
            neg_count += 1
        else:
            neu_count += 1

        row_dict = {
            "ID": item.id,
            "Input Text": item.text,
            "Processed Text": processed,
            "Sentiment": sentiment,
            "Positive": scores["pos"],
            "Negative": scores["neg"],
            "Neutral": scores["neu"],
            "Compound": scores["compound"],
            "Status": "Valid",
            "Error": None,
        }

        # Emotion Analysis (Milestone 2)
        if include_emotion and classifier is not None and classifier.is_loaded:
            # Transformer uses the preserved original text for rich context
            emotion_pred = classifier.predict(item.text, threshold=emotion_threshold)
            primary_emo = emotion_pred.primary_emotion
            primary_conf = emotion_pred.primary_confidence
            
            detected_list = [f"{d['emotion']} ({d['confidence_pct']})" for d in emotion_pred.detected_emotions]
            detected_str = ", ".join(detected_list) if detected_list else f"None (Primary: {primary_emo})"

            if primary_emo in emotion_counts:
                emotion_counts[primary_emo] += 1

            combined_analysis = f"{sentiment} Sentiment | Primary Emotion: {primary_emo} ({primary_conf * 100:.1f}%)"
            if detected_list:
                combined_analysis += f" | Detected: {', '.join([d['emotion'] for d in emotion_pred.detected_emotions])}"

            row_dict.update({
                "Emotion Model": emotion_model_type.upper(),
                "Primary Emotion": primary_emo,
                "Primary Confidence": round(primary_conf, 4),
                "Detected Emotions": detected_str,
                "Combined Analysis": combined_analysis,
            })

            # Add all 6 individual probabilities
            for emo in EMOTIONS:
                display_name = EMOTION_DISPLAY_NAMES[emo]
                row_dict[f"Prob_{display_name}"] = round(emotion_pred.probabilities.get(display_name, 0.0), 4)

        elif include_emotion:
            row_dict.update({
                "Emotion Model": emotion_model_type.upper(),
                "Primary Emotion": "Model Not Loaded",
                "Primary Confidence": None,
                "Detected Emotions": "N/A",
                "Combined Analysis": f"{sentiment} Sentiment (Emotion Model Not Loaded)",
            })
            for emo in EMOTIONS:
                row_dict[f"Prob_{EMOTION_DISPLAY_NAMES[emo]}"] = None

        records.append(row_dict)

    report_df = pd.DataFrame(records)

    summary_stats = {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "analyzed_samples": analyzed_samples,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "emotion_counts": emotion_counts if include_emotion else None,
    }

    return report_df, summary_stats


def generate_sentiment_report(
    items: List[IngestedItem],
    analyze_source_text: str = "original",
) -> pd.DataFrame:
    """Milestone 1 convenience function returning only the DataFrame report."""
    df, _ = process_pipeline_items(items, analyze_source_text=analyze_source_text, include_emotion=False)
    return df


def generate_complete_report(
    items: List[IngestedItem],
    emotion_model_type: str = "bert",
    emotion_threshold: float = DEFAULT_THRESHOLD,
    analyze_source_text: str = "original",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Milestone 1 + 2 integrated report function."""
    return process_pipeline_items(
        items,
        analyze_source_text=analyze_source_text,
        include_emotion=True,
        emotion_model_type=emotion_model_type,
        emotion_threshold=emotion_threshold,
    )


def calculate_summary_stats(items: List[IngestedItem]) -> Dict[str, Any]:
    """Convenience function returning only the summary statistics dictionary."""
    _, stats = process_pipeline_items(items)
    return stats
