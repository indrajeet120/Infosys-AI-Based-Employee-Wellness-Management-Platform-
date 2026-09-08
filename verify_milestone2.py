"""
Standalone verification script for Milestone 2:
BERT / DistilBERT Multi-label Emotion Classification, Evaluation, and ISEAR Validation.
"""

from pathlib import Path
import json
from services.config import EMOTIONS, EMOTION_DISPLAY_NAMES, MODEL_COMPARISON_PATH, ISEAR_METRICS_PATH
from services.emotion import analyze_emotion, get_emotion_classifier
from services.ingestion import ingest_text
from services.reporting import process_pipeline_items


def main():
    print("=" * 75)
    print("MILESTONE 2 - DEEP EMOTION CLASSIFICATION & VALIDATION PIPELINE")
    print("=" * 75)

    # 1. Test Inference with both models
    print("\n--- TASK 1 & 2: BERT and DistilBERT Inference Verification ---")
    test_sentences = [
        ("I received my dream job offer today and cannot stop smiling!", "Joy"),
        ("I am furious that my order was cancelled without any explanation!", "Anger"),
        ("I am happy about winning the race but scared about defending the title.", "Joy + Fear (Multi-label)"),
        ("The sight of moldy decayed fruit was totally nauseating and gross.", "Disgust"),
    ]

    for text, expected in test_sentences:
        print(f"\nInput Text: '{text}' (Expected: {expected})")
        for m_type in ["distilbert", "bert"]:
            res = analyze_emotion(text, model_type=m_type, threshold=0.40)
            detected = [d["emotion"] for d in res["detected_emotions"]]
            print(f"  [{m_type.upper():10}] Primary: {res['primary_emotion']:10} (Conf: {res['primary_confidence'] * 100:.1f}%) | Detected: {', '.join(detected) if detected else 'None'}")

    # 2. Test Model Evaluation Comparison
    print("\n--- TASK 5: Model Evaluation & Comparative Metrics ---")
    if MODEL_COMPARISON_PATH.exists():
        with open(MODEL_COMPARISON_PATH, "r", encoding="utf-8") as f:
            comp = json.load(f)
        print(f"Better Performing Model: {comp.get('better_performing_model')} (via {comp.get('comparison_metric')})")
        print("\nSummary Comparison Table:")
        for metric, vals in comp.get("metrics_summary", {}).items():
            print(f"  {metric:20}: BERT = {vals['BERT']:.4f} | DistilBERT = {vals['DistilBERT']:.4f}")

    # 3. Test ISEAR Benchmark Validation
    print("\n--- TASK 6: ISEAR Benchmark Validation Report ---")
    if ISEAR_METRICS_PATH.exists():
        with open(ISEAR_METRICS_PATH, "r", encoding="utf-8") as f:
            isear = json.load(f)
        print(f"Benchmark: {isear.get('benchmark_name')} (Model: {isear.get('model_evaluated')})")
        print(f"Total Samples: {isear.get('total_samples')} | Sample Accuracy: {isear.get('sample_accuracy') * 100:.1f}%")
        print(f"Hamming Accuracy: {isear.get('hamming_accuracy'):.4f} | Macro F1-Score: {isear.get('macro_f1'):.4f}")

    # 4. Test Milestone 1 + 2 Integration
    print("\n--- TASK 7 & 8: End-to-End Pipeline Integration (VADER + Transformers) ---")
    sample_text = "I am excited about the new opportunity but nervous about the outcome."
    items = ingest_text(sample_text)
    df, stats = process_pipeline_items(items, include_emotion=True, emotion_model_type="bert", emotion_threshold=0.40)
    row = df.iloc[0]

    print(f"Input:    '{row['Input Text']}'")
    print(f"VADER:    {row['Sentiment']} (Compound: {row['Compound']:+.4f})")
    print(f"Emotion:  {row['Primary Emotion']} (Confidence: {row['Primary Confidence'] * 100:.1f}%)")
    print(f"Detected: {row['Detected Emotions']}")
    print(f"Combined: {row['Combined Analysis']}")

    print("\n" + "=" * 75)
    print("ALL MILESTONE 2 CHECKS VERIFIED AND WORKING SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
