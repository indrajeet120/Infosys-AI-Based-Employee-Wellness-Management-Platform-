"""
Model evaluation module for Multi-Label Emotion Classification.
Computes multi-label Accuracy, Precision, Recall, Macro F1, and comparative metrics for BERT and DistilBERT.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from services.config import (
    EMOTIONS,
    EMOTION_DISPLAY_NAMES,
    REPORTS_DIR,
    BERT_METRICS_PATH,
    DISTILBERT_METRICS_PATH,
    MODEL_COMPARISON_PATH,
    TEST_DATA_PATH,
    DEFAULT_THRESHOLD,
)
from services.dataset_loader import load_emotion_dataframe
from services.emotion import EmotionClassifier, get_emotion_classifier


def evaluate_emotion_model(
    classifier_or_type: Union[str, EmotionClassifier],
    test_data_path: Union[str, Path] = TEST_DATA_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    save_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluates an emotion classification model on a test dataset.

    Returns:
        Dictionary containing overall metrics and per-emotion performance.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if isinstance(classifier_or_type, str):
        classifier = get_emotion_classifier(classifier_or_type)
    else:
        classifier = classifier_or_type

    if not classifier.is_loaded:
        raise RuntimeError(
            f"Cannot evaluate {classifier.model_type.upper()}: Model is not loaded. Train the model first."
        )

    # 1. Load ground truth
    df_test = load_emotion_dataframe(test_data_path)
    texts = df_test["text"].tolist()
    y_true = df_test[EMOTIONS].values.astype(np.float32)

    # 2. Run batch predictions
    y_pred = []
    y_probs = []

    for text in texts:
        pred = classifier.predict(text, threshold=threshold)
        prob_vector = [pred.probabilities[EMOTION_DISPLAY_NAMES[emo]] for emo in EMOTIONS]
        pred_vector = [1.0 if p >= threshold else 0.0 for p in prob_vector]
        y_probs.append(prob_vector)
        y_pred.append(pred_vector)

    y_pred = np.array(y_pred, dtype=np.float32)
    y_probs = np.array(y_probs, dtype=np.float32)

    # 3. Calculate multi-label metrics
    # Exact Match Ratio (Subset Accuracy)
    subset_accuracy = float(accuracy_score(y_true, y_pred))
    # Hamming Accuracy (1 - Hamming Loss: fraction of labels correctly predicted)
    h_loss = float(hamming_loss(y_true, y_pred))
    hamming_accuracy = float(1.0 - h_loss)

    # Macro & Micro Metrics
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    micro_precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    micro_recall = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    # Emotion-wise Breakdown
    emotion_metrics = {}
    for idx, emo in enumerate(EMOTIONS):
        display_name = EMOTION_DISPLAY_NAMES[emo]
        y_t_emo = y_true[:, idx]
        y_p_emo = y_pred[:, idx]
        
        p = float(precision_score(y_t_emo, y_p_emo, zero_division=0))
        r = float(recall_score(y_t_emo, y_p_emo, zero_division=0))
        f = float(f1_score(y_t_emo, y_p_emo, zero_division=0))
        acc = float(accuracy_score(y_t_emo, y_p_emo))
        
        emotion_metrics[display_name] = {
            "accuracy": round(acc, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": int(np.sum(y_t_emo)),
        }

    results = {
        "model": classifier.model_type.upper(),
        "total_test_samples": len(texts),
        "threshold": threshold,
        "exact_match_accuracy": round(subset_accuracy, 4),
        "hamming_accuracy": round(hamming_accuracy, 4),
        "accuracy": round(hamming_accuracy, 4),  # Standard multi-label accuracy
        "precision": round(macro_precision, 4),
        "recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "emotion_metrics": emotion_metrics,
    }

    # Save to json report if path provided or default
    if save_report_path:
        with open(save_report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {results['model']} evaluation metrics to: {save_report_path}")

    return results


def compare_models(
    test_data_path: Union[str, Path] = TEST_DATA_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """
    Evaluates both BERT and DistilBERT and generates a side-by-side comparison.

    Returns:
        Comparative dictionary and saves reports/model_comparison.json.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    bert_results = evaluate_emotion_model("bert", test_data_path=test_data_path, threshold=threshold, save_report_path=BERT_METRICS_PATH)
    distilbert_results = evaluate_emotion_model("distilbert", test_data_path=test_data_path, threshold=threshold, save_report_path=DISTILBERT_METRICS_PATH)

    better_model = "BERT" if bert_results["macro_f1"] >= distilbert_results["macro_f1"] else "DistilBERT"

    comparison = {
        "comparison_metric": "Macro F1-Score",
        "better_performing_model": better_model,
        "metrics_summary": {
            "Accuracy": {
                "BERT": bert_results["accuracy"],
                "DistilBERT": distilbert_results["accuracy"],
            },
            "Precision": {
                "BERT": bert_results["precision"],
                "DistilBERT": distilbert_results["precision"],
            },
            "Recall": {
                "BERT": bert_results["recall"],
                "DistilBERT": distilbert_results["recall"],
            },
            "Macro F1": {
                "BERT": bert_results["macro_f1"],
                "DistilBERT": distilbert_results["macro_f1"],
            },
            "Exact Match Ratio": {
                "BERT": bert_results["exact_match_accuracy"],
                "DistilBERT": distilbert_results["exact_match_accuracy"],
            },
        },
        "bert_details": bert_results,
        "distilbert_details": distilbert_results,
    }

    with open(MODEL_COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n=======================================================")
    print("MODEL COMPARISON RESULTS:")
    print(f"BERT Macro F1:       {bert_results['macro_f1']:.4f} | Accuracy: {bert_results['accuracy']:.4f}")
    print(f"DistilBERT Macro F1: {distilbert_results['macro_f1']:.4f} | Accuracy: {distilbert_results['accuracy']:.4f}")
    print(f"Better Performing Model: {better_model}")
    print(f"Saved comparison to: {MODEL_COMPARISON_PATH}")
    print("=======================================================\n")

    return comparison
