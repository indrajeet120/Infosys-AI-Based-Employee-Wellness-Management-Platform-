"""
Training script for DistilBERT Multi-Label Emotion Classifier.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.config import (
    DISTILBERT_BASE_MODEL,
    DISTILBERT_MODEL_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
)
from services.model_training import train_emotion_model


def main():
    print("Starting DistilBERT training script...")
    train_emotion_model(
        model_name_or_path=DISTILBERT_BASE_MODEL,
        output_dir=DISTILBERT_MODEL_DIR,
        train_data_path=TRAIN_DATA_PATH,
        val_data_path=VAL_DATA_PATH,
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        learning_rate=DEFAULT_LEARNING_RATE,
    )


if __name__ == "__main__":
    main()
