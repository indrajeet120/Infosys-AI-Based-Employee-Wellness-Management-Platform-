"""
Integration tests for the complete Milestone 1 end-to-end pipeline:
Ingestion -> Validation -> Preprocessing -> VADER -> Classification -> Report
"""

from io import StringIO
from pathlib import Path
import pytest
from services.ingestion import ingest_text, ingest_txt_file, ingest_csv_file
from services.reporting import process_pipeline_items, generate_sentiment_report, calculate_summary_stats


class TestPipelineIntegration:
    def test_pipeline_direct_positive_input(self):
        # Direct positive input: "I love this application!"
        raw_text = "I love this application!"
        items = ingest_text(raw_text)
        assert len(items) == 1
        assert items[0].is_valid is True

        df, stats = process_pipeline_items(items)
        assert len(df) == 1
        assert df.iloc[0]["Sentiment"] == "Positive"
        assert df.iloc[0]["Input Text"] == raw_text
        assert len(df.iloc[0]["Processed Text"]) > 0
        assert df.iloc[0]["Compound"] >= 0.05
        assert stats["valid_samples"] == 1
        assert stats["positive_count"] == 1
        assert stats["analyzed_samples"] == 1

    def test_pipeline_direct_negative_input(self):
        # Direct negative input: "This application is terrible."
        raw_text = "This application is terrible."
        items = ingest_text(raw_text)
        df, stats = process_pipeline_items(items)
        assert len(df) == 1
        assert df.iloc[0]["Sentiment"] == "Negative"
        assert df.iloc[0]["Input Text"] == raw_text
        assert df.iloc[0]["Compound"] <= -0.05
        assert stats["negative_count"] == 1

    def test_pipeline_direct_neutral_input(self):
        # Direct neutral input: "The application was installed yesterday."
        raw_text = "The application was installed yesterday."
        items = ingest_text(raw_text)
        df, stats = process_pipeline_items(items)
        assert len(df) == 1
        assert df.iloc[0]["Sentiment"] == "Neutral"
        assert df.iloc[0]["Input Text"] == raw_text
        assert -0.05 < df.iloc[0]["Compound"] < 0.05
        assert stats["neutral_count"] == 1

    def test_pipeline_txt_upload(self):
        # TXT file with positive, negative, and neutral lines
        txt_content = (
            "I love this product.\n"
            "This service is terrible.\n"
            "The product was delivered today."
        )
        items = ingest_txt_file(StringIO(txt_content))
        assert len(items) == 3
        assert all(item.is_valid for item in items)

        df, stats = process_pipeline_items(items)
        assert len(df) == 3
        assert stats["total_samples"] == 3
        assert stats["valid_samples"] == 3
        assert stats["analyzed_samples"] == 3
        assert stats["positive_count"] == 1
        assert stats["negative_count"] == 1
        assert stats["neutral_count"] == 1

    def test_pipeline_csv_upload(self):
        csv_content = (
            "text\n"
            "I love this product.\n"
            "This product is terrible.\n"
            "The product was delivered today."
        )
        items = ingest_csv_file(StringIO(csv_content))
        assert len(items) == 3

        df, stats = process_pipeline_items(items)
        assert len(df) == 3
        assert df.iloc[0]["Sentiment"] == "Positive"
        assert df.iloc[1]["Sentiment"] == "Negative"
        assert df.iloc[2]["Sentiment"] == "Neutral"
        assert stats["analyzed_samples"] == 3
        assert stats["positive_count"] == 1
        assert stats["negative_count"] == 1
        assert stats["neutral_count"] == 1

    def test_pipeline_invalid_inputs(self):
        # 1. Empty text
        empty_items = ingest_text("")
        df_empty, stats_empty = process_pipeline_items(empty_items)
        assert df_empty.iloc[0]["Status"] == "Invalid"
        assert stats_empty["invalid_samples"] == 1
        assert stats_empty["analyzed_samples"] == 0

        # 2. Whitespace text
        ws_items = ingest_text("     ")
        df_ws, stats_ws = process_pipeline_items(ws_items)
        assert df_ws.iloc[0]["Status"] == "Invalid"
        assert stats_ws["invalid_samples"] == 1

        # 3. Empty TXT
        txt_empty = ingest_txt_file(StringIO(""))
        df_te, stats_te = process_pipeline_items(txt_empty)
        assert df_te.iloc[0]["Status"] == "Invalid"

        # 4. Empty CSV
        csv_empty = ingest_csv_file(StringIO(""))
        df_ce, stats_ce = process_pipeline_items(csv_empty)
        assert df_ce.iloc[0]["Status"] == "Invalid"

        # 5. CSV without text column
        csv_no_text = ingest_csv_file(StringIO("id,review\n1,Good"))
        df_no_col, stats_no_col = process_pipeline_items(csv_no_text)
        assert df_no_col.iloc[0]["Status"] == "Invalid"

    def test_sample_corpus_file_benchmark(self):
        corpus_path = Path(__file__).resolve().parent.parent / "data" / "sample_corpus.csv"
        assert corpus_path.exists(), f"Missing sample corpus at {corpus_path}"

        items = ingest_csv_file(corpus_path)
        assert len(items) == 9

        df, stats = process_pipeline_items(items)
        assert stats["total_samples"] == 9
        assert stats["valid_samples"] == 9
        assert stats["invalid_samples"] == 0
        assert stats["analyzed_samples"] == 9
        assert stats["positive_count"] + stats["negative_count"] + stats["neutral_count"] == 9
        assert stats["positive_count"] >= 3
        assert stats["negative_count"] >= 3
        assert stats["neutral_count"] >= 2

        # Verify required report columns are present
        required_cols = [
            "ID", "Input Text", "Processed Text", "Sentiment",
            "Positive", "Negative", "Neutral", "Compound"
        ]
        for col in required_cols:
            assert col in df.columns
