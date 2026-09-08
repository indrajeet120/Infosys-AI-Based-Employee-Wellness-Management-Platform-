"""
ISEAR Benchmark Validation Service.
Validates emotion models on a held-out ISEAR benchmark dataset,
calculates emotion-wise metrics, identifies prediction discrepancies, and saves evaluation reports.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from services.config import (
    EMOTIONS,
    EMOTION_DISPLAY_NAMES,
    REPORTS_DIR,
    ISEAR_BENCHMARK_PATH,
    ISEAR_RESULTS_PATH,
    ISEAR_METRICS_PATH,
    DEFAULT_THRESHOLD,
)
from services.dataset_loader import load_emotion_dataframe
from services.emotion import EmotionClassifier, get_emotion_classifier


def validate_on_isear_benchmark(
    classifier_or_type: Union[str, EmotionClassifier] = "bert",
    isear_data_path: Union[str, Path] = ISEAR_BENCHMARK_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """
    Executes benchmark validation against the held-out ISEAR dataset.

    Returns:
        Dictionary of ISEAR evaluation metrics and per-sample results.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if isinstance(classifier_or_type, str):
        classifier = get_emotion_classifier(classifier_or_type)
    else:
        classifier = classifier_or_type

    if not classifier.is_loaded:
        raise RuntimeError(
            f"Cannot validate ISEAR benchmark: {classifier.model_type.upper()} model is not loaded."
        )

    # 1. Load ISEAR dataset
    df_isear = load_emotion_dataframe(isear_data_path)
    texts = df_isear["text"].tolist()
    y_true = df_isear[EMOTIONS].values.astype(np.float32)

    # 2. Run predictions
    sample_records = []
    y_pred = []
    y_probs = []

    correct_samples_count = 0

    for idx, text in enumerate(texts):
        pred = classifier.predict(text, threshold=threshold)
        
        # Ground truth emotions list
        expected_emotions = [
            EMOTION_DISPLAY_NAMES[emo]
            for emo_idx, emo in enumerate(EMOTIONS)
            if y_true[idx, emo_idx] > 0.5
        ]
        
        # Detected emotions list
        predicted_emotions = [d["emotion"] for d in pred.detected_emotions]
        if not predicted_emotions:
            # If no emotion met threshold, use primary emotion as prediction
            predicted_emotions = [pred.primary_emotion]

        prob_vector = [pred.probabilities[EMOTION_DISPLAY_NAMES[emo]] for emo in EMOTIONS]
        pred_binary_vector = [1.0 if p >= threshold else 0.0 for p in prob_vector]
        
        # Exact match or primary match check
        is_exact_match = (set(expected_emotions) == set([d["emotion"] for d in pred.detected_emotions]))
        is_primary_correct = pred.primary_emotion in expected_emotions

        is_correct = is_exact_match or is_primary_correct
        if is_correct:
            correct_samples_count += 1

        sample_records.append({
            "Sample_ID": idx + 1,
            "Text": text,
            "Expected_Emotion": ", ".join(expected_emotions),
            "Primary_Predicted": pred.primary_emotion,
            "Predicted_Emotions": ", ".join(predicted_emotions),
            "Primary_Confidence": round(pred.primary_confidence, 4),
            "Is_Correct": "Correct" if is_correct else "Incorrect",
            "Exact_Match": "Yes" if is_exact_match else "No",
        })

        y_probs.append(prob_vector)
        y_pred.append(pred_binary_vector)

    y_pred = np.array(y_pred, dtype=np.float32)
    y_probs = np.array(y_probs, dtype=np.float32)

    # 3. Compute Metrics
    h_loss = float(hamming_loss(y_true, y_pred))
    hamming_acc = float(1.0 - h_loss)
    sample_accuracy = float(correct_samples_count / len(texts))
    
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Emotion-wise Breakdown
    emotion_breakdown = {}
    for idx, emo in enumerate(EMOTIONS):
        display_name = EMOTION_DISPLAY_NAMES[emo]
        y_t_emo = y_true[:, idx]
        y_p_emo = y_pred[:, idx]
        
        p = float(precision_score(y_t_emo, y_p_emo, zero_division=0))
        r = float(recall_score(y_t_emo, y_p_emo, zero_division=0))
        f = float(f1_score(y_t_emo, y_p_emo, zero_division=0))
        acc = float(accuracy_score(y_t_emo, y_p_emo))
        
        emotion_breakdown[display_name] = {
            "accuracy": round(acc, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": int(np.sum(y_t_emo)),
        }

    # Save detailed CSV results
    results_df = pd.DataFrame(sample_records)
    results_df.to_csv(ISEAR_RESULTS_PATH, index=False)

    incorrect_predictions = results_df[results_df["Is_Correct"] == "Incorrect"].to_dict(orient="records")

    summary_metrics = {
        "benchmark_name": "ISEAR Held-Out Subset",
        "model_evaluated": classifier.model_type.upper(),
        "total_samples": len(texts),
        "correct_predictions_count": correct_samples_count,
        "incorrect_predictions_count": len(texts) - correct_samples_count,
        "sample_accuracy": round(sample_accuracy, 4),
        "hamming_accuracy": round(hamming_acc, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "emotion_wise_performance": emotion_breakdown,
        "incorrect_samples_count": len(incorrect_predictions),
    }

    with open(ISEAR_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"\n=======================================================")
    print(f"ISEAR BENCHMARK VALIDATION RESULTS ({classifier.model_type.upper()}):")
    print(f"Total Samples: {len(texts)} | Correct: {correct_samples_count} | Sample Accuracy: {sample_accuracy * 100:.1f}%")
    print(f"Hamming Accuracy: {hamming_acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(f"Saved CSV Report: {ISEAR_RESULTS_PATH}")
    print(f"Saved JSON Metrics: {ISEAR_METRICS_PATH}")
    print("=======================================================\n")

    return {
        "metrics": summary_metrics,
        "results_df": results_df,
        "incorrect_predictions": incorrect_predictions,
    }
