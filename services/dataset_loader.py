"""
Dataset loader and preparation module for Multi-Label Emotion Classification.
Supports converting diverse emotion formats into multi-label binary vectors for the 6 core emotions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from services.config import EMOTIONS, EMOTION2ID, NUM_EMOTIONS, MAX_SEQ_LENGTH

# Emotion mapping dictionary to handle synonymous or dataset-specific variations
LABEL_NORMALIZATION_MAP = {
    # Joy
    "joy": "joy",
    "happy": "joy",
    "happiness": "joy",
    "excited": "joy",
    "excitement": "joy",
    "love": "joy",
    "delight": "joy",
    "pleasure": "joy",
    "pride": "joy",
    "relief": "joy",
    "admiration": "joy",
    "amusement": "joy",
    "gratitude": "joy",
    # Sadness
    "sadness": "sadness",
    "sad": "sadness",
    "sorrow": "sadness",
    "grief": "sadness",
    "depressed": "sadness",
    "disappointment": "sadness",
    "embarrassment": "sadness",
    "remorse": "sadness",
    # Anger
    "anger": "anger",
    "angry": "anger",
    "rage": "anger",
    "furious": "anger",
    "irritation": "anger",
    "annoyance": "anger",
    # Fear
    "fear": "fear",
    "scared": "fear",
    "afraid": "fear",
    "anxiety": "fear",
    "nervous": "fear",
    "panic": "fear",
    "terror": "fear",
    "worry": "fear",
    # Surprise
    "surprise": "surprise",
    "surprised": "surprise",
    "shock": "surprise",
    "astonishment": "surprise",
    "curiosity": "surprise",
    "confusion": "surprise",
    # Disgust
    "disgust": "disgust",
    "disgusted": "disgust",
    "revulsion": "disgust",
    "loathing": "disgust",
    "shame": "disgust",
    "guilt": "disgust",
}


def normalize_emotion_label(label: str) -> Optional[str]:
    """Normalizes an emotion string to one of the 6 core emotions or None."""
    cleaned = str(label).strip().lower()
    return LABEL_NORMALIZATION_MAP.get(cleaned, None)


def parse_labels_to_vector(
    row_or_val: Union[pd.Series, str, List[str]],
    emotions: List[str] = EMOTIONS
) -> np.ndarray:
    """
    Converts a row or label representation into a binary multi-label vector.
    
    Returns:
        np.ndarray of shape (6,) with dtype float32.
    """
    vec = np.zeros(len(emotions), dtype=np.float32)
    
    if isinstance(row_or_val, pd.Series):
        # Check if individual emotion columns exist in series
        has_all_cols = all(emo in row_or_val for emo in emotions)
        if has_all_cols:
            for idx, emo in enumerate(emotions):
                val = row_or_val[emo]
                vec[idx] = 1.0 if (not pd.isna(val) and float(val) > 0.5) else 0.0
            return vec
        
        # Check for label/emotion/emotions column
        for col_name in ["emotion", "emotions", "label", "labels", "Emotion", "Labels"]:
            if col_name in row_or_val and not pd.isna(row_or_val[col_name]):
                return parse_labels_to_vector(str(row_or_val[col_name]), emotions)
        return vec

    if isinstance(row_or_val, list):
        for item in row_or_val:
            norm = normalize_emotion_label(item)
            if norm in EMOTION2ID:
                vec[EMOTION2ID[norm]] = 1.0
        return vec

    if isinstance(row_or_val, str):
        # Split by comma, pipe, semicolon, or plus
        raw_parts = [p.strip() for p in row_or_val.replace("|", ",").replace(";", ",").replace("+", ",").split(",") if p.strip()]
        for part in raw_parts:
            norm = normalize_emotion_label(part)
            if norm in EMOTION2ID:
                vec[EMOTION2ID[norm]] = 1.0

    return vec


def load_emotion_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads and standardizes an emotion dataset from CSV.
    Ensures 'text' column and binary columns for all 6 emotions exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Emotion dataset not found at: {path}")

    df = pd.read_csv(path)
    
    # Locate text column
    text_col = None
    for candidate in ["text", "Text", "sentence", "Sentence", "content", "Content"]:
        if candidate in df.columns:
            text_col = candidate
            break
            
    if text_col is None:
        raise ValueError(f"Dataset at {path} missing a 'text' column. Columns: {list(df.columns)}")

    # Rename text column to 'text'
    if text_col != "text":
        df = df.rename(columns={text_col: "text"})

    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()

    # Check if 6 binary columns already exist
    has_all_emo_cols = all(emo in df.columns for emo in EMOTIONS)
    if not has_all_emo_cols:
        vectors = []
        for _, row in df.iterrows():
            vec = parse_labels_to_vector(row)
            vectors.append(vec)
        vectors_arr = np.array(vectors, dtype=np.float32)
        for idx, emo in enumerate(EMOTIONS):
            df[emo] = vectors_arr[:, idx]

    return df


class MultiLabelEmotionDataset(Dataset):
    """
    PyTorch Dataset for multi-label emotion classification with Hugging Face Tokenizers.
    """

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer: any,
        max_length: int = MAX_SEQ_LENGTH,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        
        # Token type IDs for BERT if present
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
            
        return item
