"""
Model training service for Multi-Label Emotion Classification using BERT and DistilBERT.
Implements fine-tuning with BCEWithLogitsLoss, validation tracking, and model artifact saving.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from services.config import (
    EMOTIONS,
    EMOTION2ID,
    ID2EMOTION,
    NUM_EMOTIONS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS,
    MAX_SEQ_LENGTH,
    get_device,
)
from services.dataset_loader import (
    MultiLabelEmotionDataset,
    load_emotion_dataframe,
)


def train_emotion_model(
    model_name_or_path: str,
    output_dir: Path,
    train_data_path: Path,
    val_data_path: Optional[Path] = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Fine-tunes a Transformer model for 6-emotion multi-label classification.

    Args:
        model_name_or_path: Hugging Face model identifier (e.g. 'bert-base-uncased' or 'distilbert-base-uncased').
        output_dir: Directory where model and tokenizer artifacts will be saved.
        train_data_path: CSV path containing training samples.
        val_data_path: Optional CSV path containing validation samples.
        epochs: Number of training epochs.
        batch_size: DataLoader batch size.
        learning_rate: Learning rate for AdamW optimizer.
        device: Torch device (defaults to get_device()).

    Returns:
        Dictionary containing training history and final validation metrics.
    """
    if device is None:
        device = get_device()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"Starting Multi-Label Emotion Fine-Tuning: {model_name_or_path}")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print(f"Target Output Directory: {output_dir}")
    print(f"=======================================================\n")

    # 1. Load Data
    df_train = load_emotion_dataframe(train_data_path)
    train_texts = df_train["text"].tolist()
    train_labels = df_train[EMOTIONS].values.astype(np.float32)

    val_texts, val_labels = [], None
    if val_data_path and Path(val_data_path).exists():
        df_val = load_emotion_dataframe(val_data_path)
        val_texts = df_val["text"].tolist()
        val_labels = df_val[EMOTIONS].values.astype(np.float32)

    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=NUM_EMOTIONS,
        problem_type="multi_label_classification",
        id2label=ID2EMOTION,
        label2id=EMOTION2ID,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )
    model.to(device)

    # 3. DataLoaders
    train_dataset = MultiLabelEmotionDataset(train_texts, train_labels, tokenizer, max_length=MAX_SEQ_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_labels is not None and len(val_texts) > 0:
        val_dataset = MultiLabelEmotionDataset(val_texts, val_labels, tokenizer, max_length=MAX_SEQ_LENGTH)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Optimizer & Dynamic Pos-Weighted Loss for Multi-Label Balance
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Calculate positive weight per emotion to balance multi-label imbalance
    pos_counts = np.sum(train_labels, axis=0)
    total_samples = len(train_labels)
    neg_counts = total_samples - pos_counts
    pos_weights = np.clip(neg_counts / np.maximum(pos_counts, 1.0), 1.0, 10.0)
    pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    history = {"train_loss": [], "val_loss": [], "epochs": epochs}
    best_val_loss = float("inf")

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / max(num_batches, 1)
        history["train_loss"].append(round(avg_train_loss, 4))

        val_msg = ""
        if val_loader:
            model.eval()
            total_val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    val_loss = criterion(outputs.logits, labels)
                    total_val_loss += val_loss.item()
                    val_batches += 1

            avg_val_loss = total_val_loss / max(val_batches, 1)
            history["val_loss"].append(round(avg_val_loss, 4))
            val_msg = f" | Val Loss: {avg_val_loss:.4f}"

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f}{val_msg}")

    # 6. Save Model, Tokenizer, and Training Metadata
    print(f"\nSaving fine-tuned model and tokenizer to: {output_dir} ...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    training_summary = {
        "model_architecture": model_name_or_path,
        "emotions": EMOTIONS,
        "num_emotions": NUM_EMOTIONS,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "history": history,
    }

    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    print(f"Training successfully completed and artifacts saved to {output_dir}\n")
    return training_summary
