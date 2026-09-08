"""
Emotion classification inference service for BERT and DistilBERT models.
Computes multi-label sigmoid probabilities, dynamic confidence scores,
threshold-based detected emotions, and primary emotion identification.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from services.config import (
    EMOTIONS,
    EMOTION_DISPLAY_NAMES,
    BERT_MODEL_DIR,
    DISTILBERT_MODEL_DIR,
    DEFAULT_THRESHOLD,
    MAX_SEQ_LENGTH,
    get_device,
)


@dataclass
class EmotionPrediction:
    """Structured container for emotion classification predictions."""
    model_type: str
    primary_emotion: str
    primary_confidence: float
    detected_emotions: List[Dict[str, Union[str, float]]]
    probabilities: Dict[str, float]
    threshold: float
    is_valid: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "primary_emotion": self.primary_emotion,
            "primary_confidence": round(self.primary_confidence, 4),
            "detected_emotions": self.detected_emotions,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "threshold": self.threshold,
            "is_valid": self.is_valid,
            "error_message": self.error_message,
        }


class EmotionClassifier:
    """
    Multi-label Transformer Emotion Classifier for 6 core emotions.
    """

    def __init__(
        self,
        model_type: str = "bert",
        model_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.model_type = model_type.lower()
        self.device = device if device is not None else get_device()
        
        if model_dir is not None:
            self.model_dir = Path(model_dir)
        elif self.model_type in ["distilbert", "distilbert_emotion"]:
            self.model_dir = DISTILBERT_MODEL_DIR
            self.model_type = "distilbert"
        else:
            self.model_dir = BERT_MODEL_DIR
            self.model_type = "bert"

        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        """Loads model and tokenizer from model_dir if present."""
        if not self.model_dir.exists():
            print(f"Notice: Model directory not found at {self.model_dir}. Run training script to create it.")
            self.is_loaded = False
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"Successfully loaded {self.model_type.upper()} emotion model from {self.model_dir} on {self.device}")
        except Exception as e:
            print(f"Warning: Failed to load model from {self.model_dir}: {e}")
            self.is_loaded = False

    def predict(
        self,
        text: Optional[str],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> EmotionPrediction:
        """
        Runs multi-label emotion prediction on input text.

        Returns:
            EmotionPrediction object with dynamic probabilities and detected labels.
        """
        model_display = "BERT" if self.model_type == "bert" else "DistilBERT"
        
        # Validation checks
        if text is None:
            return EmotionPrediction(
                model_type=model_display,
                primary_emotion="None",
                primary_confidence=0.0,
                detected_emotions=[],
                probabilities={EMOTION_DISPLAY_NAMES[e]: 0.0 for e in EMOTIONS},
                threshold=threshold,
                is_valid=False,
                error_message="Input text is null/None.",
            )

        text_str = str(text).strip()
        if not text_str:
            return EmotionPrediction(
                model_type=model_display,
                primary_emotion="None",
                primary_confidence=0.0,
                detected_emotions=[],
                probabilities={EMOTION_DISPLAY_NAMES[e]: 0.0 for e in EMOTIONS},
                threshold=threshold,
                is_valid=False,
                error_message="Input text is empty or whitespace-only.",
            )

        if not self.is_loaded or self.model is None or self.tokenizer is None:
            return EmotionPrediction(
                model_type=model_display,
                primary_emotion="Model Not Loaded",
                primary_confidence=0.0,
                detected_emotions=[],
                probabilities={EMOTION_DISPLAY_NAMES[e]: 0.0 for e in EMOTIONS},
                threshold=threshold,
                is_valid=False,
                error_message=f"Model '{self.model_type}' is not loaded. Train the model first using 'python scripts/train_{self.model_type}.py'.",
            )

        # Tokenize
        try:
            inputs = self.tokenizer(
                text_str,
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(0)
                # Multi-label Sigmoid activation
                probs = torch.sigmoid(logits).cpu().numpy()

        except Exception as e:
            return EmotionPrediction(
                model_type=model_display,
                primary_emotion="Error",
                primary_confidence=0.0,
                detected_emotions=[],
                probabilities={EMOTION_DISPLAY_NAMES[e]: 0.0 for e in EMOTIONS},
                threshold=threshold,
                is_valid=False,
                error_message=f"Inference error: {str(e)}",
            )

        # Map to emotion probabilities
        prob_dict = {}
        for idx, emo in enumerate(EMOTIONS):
            display_name = EMOTION_DISPLAY_NAMES[emo]
            prob_dict[display_name] = float(probs[idx])

        # Primary emotion: Highest probability
        max_idx = int(np.argmax(probs))
        primary_emo_key = EMOTIONS[max_idx]
        primary_emotion = EMOTION_DISPLAY_NAMES[primary_emo_key]
        primary_confidence = float(probs[max_idx])

        # Detected emotions: All emotions meeting threshold
        detected = []
        for idx, emo in enumerate(EMOTIONS):
            p = float(probs[idx])
            if p >= threshold:
                detected.append({
                    "emotion": EMOTION_DISPLAY_NAMES[emo],
                    "confidence": round(p, 4),
                    "confidence_pct": f"{p * 100:.1f}%",
                })

        # Sort detected emotions by confidence descending
        detected.sort(key=lambda x: x["confidence"], reverse=True)

        return EmotionPrediction(
            model_type=model_display,
            primary_emotion=primary_emotion,
            primary_confidence=primary_confidence,
            detected_emotions=detected,
            probabilities=prob_dict,
            threshold=threshold,
            is_valid=True,
            error_message=None,
        )


# Cached singletons
_bert_classifier: Optional[EmotionClassifier] = None
_distilbert_classifier: Optional[EmotionClassifier] = None


def get_emotion_classifier(model_type: str = "bert") -> EmotionClassifier:
    """Factory function returning cached EmotionClassifier instances."""
    global _bert_classifier, _distilbert_classifier
    normalized = model_type.lower()
    
    if "distil" in normalized:
        if _distilbert_classifier is None:
            _distilbert_classifier = EmotionClassifier(model_type="distilbert")
        return _distilbert_classifier
    else:
        if _bert_classifier is None:
            _bert_classifier = EmotionClassifier(model_type="bert")
        return _bert_classifier


def analyze_emotion(
    text: str,
    model_type: str = "bert",
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Convenience function for emotion analysis returning a dictionary."""
    classifier = get_emotion_classifier(model_type=model_type)
    pred = classifier.predict(text, threshold=threshold)
    return pred.to_dict()
