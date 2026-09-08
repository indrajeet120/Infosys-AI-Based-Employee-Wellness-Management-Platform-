"""
Complete End-to-End Verification Runner for Milestone 1.
"""

from io import StringIO
from pathlib import Path
import pandas as pd
from services.ingestion import ingest_text, ingest_txt_file, ingest_csv_file
from services.preprocessing import preprocess_text
from services.sentiment import analyze_sentiment
from services.reporting import process_pipeline_items, generate_sentiment_report, calculate_summary_stats


def main():
    print("=" * 70)
    print("MILESTONE 1 - COMPREHENSIVE PIPELINE VERIFICATION")
    print("=" * 70)

    # 1. Test Ingestion of Direct Text
    print("\n--- TASK 1: Ingestion & Validation ---")
    item_direct = ingest_text("I absolutely love this product!")
    print(f"Direct text valid: {item_direct[0].is_valid} | Text: '{item_direct[0].text}'")

    item_empty = ingest_text("   ")
    print(f"Whitespace invalid: {not item_empty[0].is_valid} | Error: '{item_empty[0].error_message}'")

    # 2. Test Preprocessing
    print("\n--- TASK 2: Preprocessing ---")
    test_cases_prep = [
        ("I really love this product!", "Normal"),
        ("I   really    love    this.", "Repeated spaces"),
        ("This product is excellent!!!", "Punctuation"),
        ("Excellent @product #happy", "Special characters"),
        ("GOOD Good good", "Casing"),
        ("I am not happy.", "Negation preservation"),
    ]
    for text, label in test_cases_prep:
        proc = preprocess_text(text)
        print(f"[{label:22}] '{text}' -> '{proc}'")

    # 3. Test VADER Sentiment
    print("\n--- TASK 3: VADER Sentiment Scoring ---")
    test_cases_sent = [
        "I absolutely love this product.",
        "I hate this product.",
        "The meeting is scheduled for Monday."
    ]
    for text in test_cases_sent:
        scores = analyze_sentiment(text)
        print(f"'{text}'\n  -> Compound: {scores['compound']:+.4f} | Pos: {scores['pos']:.3f} | Neg: {scores['neg']:.3f} | Neu: {scores['neu']:.3f} | Sentiment: {scores['sentiment']}")

    # 4. Test Benchmark Sample Corpus & Reporting
    print("\n--- TASK 4 & 5: Benchmark Corpus Report & Pipeline Integration ---")
    corpus_path = Path("data/sample_corpus.csv")
    items = ingest_csv_file(corpus_path)
    df, stats = process_pipeline_items(items)

    print("\nSummary Statistics:")
    for k, v in stats.items():
        print(f"  {k:20}: {v}")

    print("\nClassification Report:")
    print(df[["ID", "Input Text", "Processed Text", "Sentiment", "Compound"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("ALL MILESTONE 1 CHECKS EXECUTED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
