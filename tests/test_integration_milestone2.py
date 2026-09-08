"""
Integration tests for Milestone 1 + Milestone 2 end-to-end pipeline:
User Input -> Ingestion -> Preprocessing -> VADER Sentiment -> BERT/DistilBERT Emotion -> Combined Report
"""

from io import StringIO
import pytest
from services.ingestion import ingest_text, ingest_csv_file
from services.reporting import generate_complete_report, process_pipeline_items


class TestMilestone2Integration:
    def test_end_to_end_positive_joy_input(self):
        text = "I received my dream job offer today and cannot stop smiling!"
        items = ingest_text(text)
        assert items[0].is_valid is True

        df, stats = process_pipeline_items(items, include_emotion=True, emotion_model_type="bert")
        assert len(df) == 1
        row = df.iloc[0]

        # Milestone 1 verification
        assert row["Sentiment"] == "Positive"
        assert row["Compound"] >= 0.05
        assert len(row["Processed Text"]) > 0

        # Milestone 2 verification
        assert row["Emotion Model"] == "BERT"
        assert row["Primary Emotion"] == "Joy"
        assert row["Primary Confidence"] > 0.4
        assert "Prob_Joy" in row
        assert "Combined Analysis" in row

    def test_end_to_end_negative_anger_input(self):
        text = "I am furious that my order was cancelled without any explanation!"
        items = ingest_text(text)
        df, stats = process_pipeline_items(items, include_emotion=True, emotion_model_type="distilbert")
        assert len(df) == 1
        row = df.iloc[0]

        assert row["Sentiment"] == "Negative"
        assert row["Compound"] <= -0.05
        assert row["Primary Emotion"] == "Anger"

    def test_end_to_end_csv_with_emotions(self):
        csv_content = (
            "text\n"
            "I absolutely love this product!\n"
            "The service was terrible and I am angry.\n"
            "The meeting is scheduled for tomorrow."
        )
        items = ingest_csv_file(StringIO(csv_content))
        assert len(items) == 3

        df, stats = process_pipeline_items(items, include_emotion=True, emotion_model_type="distilbert")
        assert len(df) == 3
        assert stats["total_samples"] == 3
        assert stats["analyzed_samples"] == 3
        assert "Prob_Joy" in df.columns
        assert "Prob_Anger" in df.columns
        assert "Primary Emotion" in df.columns

    def test_invalid_input_stops_pipeline_gracefully(self):
        items = ingest_text("     ")
        df, stats = process_pipeline_items(items, include_emotion=True, emotion_model_type="bert")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["Status"] == "Invalid"
        assert stats["invalid_samples"] == 1
        assert stats["analyzed_samples"] == 0
