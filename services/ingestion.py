"""
Ingestion service for Text Sentiment Analysis.
Handles validation and extraction of direct text, TXT files, and CSV files.
"""

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd


@dataclass
class IngestedItem:
    """Represents a single ingested text item along with its validation status."""
    id: int
    text: str
    is_valid: bool
    error_message: Optional[str] = None
    source: str = "direct"


def validate_raw_text(text: Optional[Union[str, any]]) -> tuple[bool, str, Optional[str]]:
    """
    Validates a raw text input.

    Returns:
        (is_valid, cleaned_text, error_message)
    """
    if text is None:
        return False, "", "Text is null/None."

    if not isinstance(text, str):
        if pd.isna(text):
            return False, "", "Text is missing or NaN value."
        text = str(text)

    stripped = text.strip()
    if not stripped:
        return False, "", "Text is empty or contains only whitespace."

    return True, text, None


def ingest_text(raw_text: Optional[str], source: str = "direct") -> List[IngestedItem]:
    """
    Ingests direct text input.

    Args:
        raw_text: The user-supplied raw text string.
        source: Ingestion source identifier.

    Returns:
        List containing a single IngestedItem.
    """
    is_valid, cleaned_text, error_msg = validate_raw_text(raw_text)
    return [
        IngestedItem(
            id=1,
            text=cleaned_text if is_valid else (raw_text if raw_text is not None else ""),
            is_valid=is_valid,
            error_message=error_msg,
            source=source,
        )
    ]


def ingest_txt_file(
    file_input: Union[str, Path, BytesIO, StringIO, bytes],
    filename: Optional[str] = None,
    source: str = "txt",
) -> List[IngestedItem]:
    """
    Ingests and validates a TXT file from a path, buffer, or raw bytes.

    Args:
        file_input: File path, file-like object, or bytes.
        filename: Optional name of the file to check extension.
        source: Ingestion source identifier.

    Returns:
        List of IngestedItem for the lines/content.
    """
    # Check extension if filename provided
    if filename:
        ext = Path(filename).suffix.lower()
        if ext and ext != ".txt":
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message=f"Unsupported file format: '{ext}'. Only .txt files are allowed.",
                    source=source,
                )
            ]

    # Check path if path given
    if isinstance(file_input, (str, Path)) and not isinstance(file_input, (StringIO, BytesIO)):
        path = Path(file_input)
        if path.suffix.lower() != ".txt":
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message=f"Unsupported file format: '{path.suffix}'. Only .txt files are allowed.",
                    source=source,
                )
            ]
        if not path.exists():
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message=f"File not found: '{file_input}'",
                    source=source,
                )
            ]
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message=f"Encoding or read error: {str(e)}",
                    source=source,
                )
            ]
    elif isinstance(file_input, bytes):
        try:
            content = file_input.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content = file_input.decode("latin-1")
            except Exception as e:
                return [
                    IngestedItem(
                        id=1,
                        text="",
                        is_valid=False,
                        error_message=f"Failed to decode TXT file: {str(e)}",
                        source=source,
                    )
                ]
    elif isinstance(file_input, BytesIO):
        try:
            raw_bytes = file_input.getvalue()
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content = raw_bytes.decode("latin-1")
            except Exception as e:
                return [
                    IngestedItem(
                        id=1,
                        text="",
                        is_valid=False,
                        error_message=f"Failed to decode TXT file: {str(e)}",
                        source=source,
                    )
                ]
    elif isinstance(file_input, StringIO):
        content = file_input.getvalue()
    elif hasattr(file_input, "read"):
        # Streamlit UploadedFile or file-like object
        try:
            raw = file_input.read()
            if isinstance(raw, bytes):
                content = raw.decode("utf-8", errors="replace")
            else:
                content = str(raw)
        except Exception as e:
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message=f"Failed to read TXT stream: {str(e)}",
                    source=source,
                )
            ]
    else:
        content = str(file_input)

    if not content or not content.strip():
        return [
            IngestedItem(
                id=1,
                text="",
                is_valid=False,
                error_message="TXT file is empty or contains only whitespace.",
                source=source,
            )
        ]

    # Split lines and create items
    lines = content.splitlines()
    results: List[IngestedItem] = []
    item_id = 1
    for line in lines:
        is_valid, cleaned, err = validate_raw_text(line)
        results.append(
            IngestedItem(
                id=item_id,
                text=line,
                is_valid=is_valid,
                error_message=err,
                source=source,
            )
        )
        item_id += 1

    return results


def ingest_csv_file(
    file_input: Union[str, Path, BytesIO, StringIO, bytes, pd.DataFrame],
    text_column: str = "text",
    filename: Optional[str] = None,
    source: str = "csv",
) -> List[IngestedItem]:
    """
    Ingests and validates a CSV file or DataFrame containing a 'text' column.

    Args:
        file_input: CSV path, buffer, bytes, or DataFrame.
        text_column: Expected column name containing text data (default 'text').
        filename: Optional name of the file to check extension.
        source: Ingestion source identifier.

    Returns:
        List of IngestedItem for each row.
    """
    if filename:
        ext = Path(filename).suffix.lower()
        if ext and ext != ".csv":
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message=f"Unsupported file format: '{ext}'. Only .csv files are allowed.",
                    source=source,
                )
            ]

    # Load DataFrame
    df: Optional[pd.DataFrame] = None
    if isinstance(file_input, pd.DataFrame):
        df = file_input.copy()
    elif isinstance(file_input, (str, Path)) and not isinstance(file_input, (StringIO, BytesIO)):
        path = Path(file_input)
        if path.suffix.lower() != ".csv":
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message=f"Unsupported file format: '{path.suffix}'. Only .csv files are allowed.",
                    source=source,
                )
            ]
        if not path.exists():
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message=f"File not found: '{file_input}'",
                    source=source,
                )
            ]
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message="CSV file is empty.",
                    source=source,
                )
            ]
        except Exception as e:
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message=f"CSV read/parsing error: {str(e)}",
                    source=source,
                )
            ]
    elif isinstance(file_input, bytes):
        try:
            df = pd.read_csv(BytesIO(file_input))
        except pd.errors.EmptyDataError:
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message="CSV file is empty.",
                    source=source,
                )
            ]
        except Exception as e:
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message=f"Failed to parse CSV bytes: {str(e)}",
                    source=source,
                )
            ]
    elif isinstance(file_input, (StringIO, BytesIO)):
        try:
            df = pd.read_csv(file_input)
        except pd.errors.EmptyDataError:
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message="CSV file is empty.",
                    source=source,
                )
            ]
        except Exception as e:
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message=f"Failed to parse CSV stream: {str(e)}",
                    source=source,
                )
            ]
    elif hasattr(file_input, "read"):
        try:
            df = pd.read_csv(file_input)
        except pd.errors.EmptyDataError:
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message="CSV file is empty.",
                    source=source,
                )
            ]
        except Exception as e:
            return [
                IngestedItem(
                    id=1,
                    text="",
                    is_valid=False,
                    error_message=f"Failed to parse CSV file: {str(e)}",
                    source=source,
                )
            ]
    else:
        return [
            IngestedItem(
                id=1,
                text="",
                is_valid=False,
                error_message=f"Unsupported input type for CSV ingestion: {type(file_input)}",
                source=source,
            )
        ]

    if df is None or df.empty and len(df.columns) == 0:
        return [
            IngestedItem(
                id=1,
                text="",
                is_valid=False,
                error_message="CSV file is empty.",
                source=source,
            )
        ]

    # Check for text column (case-insensitive search if exact not found)
    matched_col = None
    if text_column in df.columns:
        matched_col = text_column
    else:
        # Check case-insensitively
        for col in df.columns:
            if str(col).strip().lower() == text_column.lower():
                matched_col = col
                break

    if matched_col is None:
        return [
            IngestedItem(
                id=1,
                text="",
                is_valid=False,
                error_message=f"CSV missing required column '{text_column}'. Available columns: {list(df.columns)}",
                source=source,
            )
        ]

    if len(df) == 0:
        return [
            IngestedItem(
                id=1,
                text="",
                is_valid=False,
                error_message="CSV file contains headers but no data rows.",
                source=source,
            )
        ]

    results: List[IngestedItem] = []
    for idx, row in df.iterrows():
        raw_val = row[matched_col]
        is_valid, cleaned, err = validate_raw_text(raw_val)
        results.append(
            IngestedItem(
                id=idx + 1,
                text=str(raw_val) if not pd.isna(raw_val) else "",
                is_valid=is_valid,
                error_message=err,
                source=source,
            )
        )

    return results
