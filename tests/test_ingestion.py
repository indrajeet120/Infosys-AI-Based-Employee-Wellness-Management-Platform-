"""
Unit tests for the Text Ingestion and Input Validation module.
"""

from io import BytesIO, StringIO
import pytest
import pandas as pd
from services.ingestion import ingest_text, ingest_txt_file, ingest_csv_file, validate_raw_text


class TestDirectTextIngestion:
    def test_valid_direct_text(self):
        items = ingest_text("I absolutely love this product.")
        assert len(items) == 1
        assert items[0].is_valid is True
        assert items[0].text == "I absolutely love this product."
        assert items[0].error_message is None
        assert items[0].source == "direct"

    def test_empty_direct_text(self):
        items = ingest_text("")
        assert len(items) == 1
        assert items[0].is_valid is False
        assert "empty" in items[0].error_message.lower()

    def test_whitespace_direct_text(self):
        items = ingest_text("     \t \n ")
        assert len(items) == 1
        assert items[0].is_valid is False
        assert "whitespace" in items[0].error_message.lower()

    def test_none_direct_text(self):
        items = ingest_text(None)
        assert len(items) == 1
        assert items[0].is_valid is False
        assert items[0].error_message is not None


class TestTxtFileIngestion:
    def test_valid_txt_buffer(self):
        content = "Line one.\nLine two.\nLine three."
        buffer = StringIO(content)
        items = ingest_txt_file(buffer)
        assert len(items) == 3
        assert all(item.is_valid for item in items)
        assert items[0].text == "Line one."
        assert items[1].text == "Line two."
        assert items[2].text == "Line three."

    def test_txt_with_empty_lines(self):
        content = "Valid line.\n   \nAnother valid line."
        buffer = StringIO(content)
        items = ingest_txt_file(buffer)
        assert len(items) == 3
        assert items[0].is_valid is True
        assert items[1].is_valid is False
        assert items[2].is_valid is True

    def test_empty_txt(self):
        buffer = StringIO("")
        items = ingest_txt_file(buffer)
        assert len(items) == 1
        assert items[0].is_valid is False
        assert "empty" in items[0].error_message.lower()

    def test_whitespace_only_txt(self):
        buffer = StringIO("   \n\t  ")
        items = ingest_txt_file(buffer)
        assert len(items) == 1
        assert items[0].is_valid is False

    def test_txt_bytes_ingestion(self):
        raw_bytes = b"Testing byte ingestion."
        items = ingest_txt_file(BytesIO(raw_bytes))
        assert len(items) == 1
        assert items[0].is_valid is True
        assert items[0].text == "Testing byte ingestion."

    def test_unsupported_file_extension(self):
        items = ingest_txt_file("sample.pdf", filename="sample.pdf")
        assert len(items) == 1
        assert items[0].is_valid is False
        assert "unsupported file format" in items[0].error_message.lower()


class TestCsvFileIngestion:
    def test_valid_csv(self):
        csv_data = "text\nI love this product.\nThis service is terrible.\nThe product was delivered today."
        buffer = StringIO(csv_data)
        items = ingest_csv_file(buffer)
        assert len(items) == 3
        assert all(item.is_valid for item in items)
        assert items[0].text == "I love this product."
        assert items[1].text == "This service is terrible."
        assert items[2].text == "The product was delivered today."

    def test_csv_case_insensitive_header(self):
        csv_data = "TEXT\nAwesome service!\n"
        buffer = StringIO(csv_data)
        items = ingest_csv_file(buffer)
        assert len(items) == 1
        assert items[0].is_valid is True
        assert items[0].text == "Awesome service!"

    def test_csv_missing_text_column(self):
        csv_data = "id,comment\n1,Great\n2,Bad"
        buffer = StringIO(csv_data)
        items = ingest_csv_file(buffer)
        assert len(items) == 1
        assert items[0].is_valid is False
        assert "missing required column 'text'" in items[0].error_message.lower()

    def test_empty_csv(self):
        buffer = StringIO("")
        items = ingest_csv_file(buffer)
        assert len(items) == 1
        assert items[0].is_valid is False
        assert "empty" in items[0].error_message.lower()

    def test_csv_header_only(self):
        csv_data = "text\n"
        buffer = StringIO(csv_data)
        items = ingest_csv_file(buffer)
        assert len(items) == 1
        assert items[0].is_valid is False
        assert "no data rows" in items[0].error_message.lower()

    def test_csv_with_nan_and_empty_rows(self):
        df = pd.DataFrame({"text": ["Good item", None, "   ", "Bad item"]})
        items = ingest_csv_file(df)
        assert len(items) == 4
        assert items[0].is_valid is True
        assert items[1].is_valid is False
        assert items[2].is_valid is False
        assert items[3].is_valid is True

    def test_unsupported_csv_extension(self):
        items = ingest_csv_file("sample.json", filename="sample.json")
        assert len(items) == 1
        assert items[0].is_valid is False
        assert "unsupported file format" in items[0].error_message.lower()
