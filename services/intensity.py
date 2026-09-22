"""
Emotion Intensity and Emotional State Analysis Service.
Enhances raw multi-label emotion predictions into rich, structured emotional states
with dynamic intensity, polarity balance, mixed emotion detection, and severity tiers.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import numpy as np

from services.config import (
    EMOTIONS,
    EMOTION_DISPLAY_NAMES,
    DEFAULT_MIXED_THRESHOLD,
    SEVERITY_THRESHOLDS,
    EMOTION_SEVERITY_WEIGHTS,
)
from services.emotion import analyze_emotion, EmotionPrediction


@dataclass
class EmotionalState:
    """Structured representation of a user's analyzed emotional state."""
    dominant_emotion: str
    dominant_confidence: float
    top_emotions: List[Dict[str, Any]] = field(default_factory=list)
    emotional_intensity: float = 0.0
    positive_polarity: float = 0.0
    negative_polarity: float = 0.0
    mixed_emotion: bool = False
    emotion_severity: str = "Low"
    final_emotional_state: str = "Neutral / Baseline"
    probabilities: Dict[str, float] = field(default_factory=dict)
    raw_text: str = ""
    model_used: str = "bert"
    is_valid: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts the emotional state to a serializable dictionary."""
        return asdict(self)


def calculate_polarity(probabilities: Dict[str, float]) -> Dict[str, float]:
    """
    Computes positive and negative emotional polarities from multi-label emotion probabilities.
    Positive: Joy
    Negative: Sadness, Anger, Fear, Disgust
    Surprise is treated as valence-neutral / contextual.
    """
    pos_score = float(probabilities.get("Joy", 0.0))
    neg_emotions = ["Sadness", "Anger", "Fear", "Disgust"]
    neg_scores = [probabilities.get(e, 0.0) for e in neg_emotions]
    
    # Negative polarity combines max negative emotion and average negative signal
    max_neg = float(max(neg_scores)) if neg_scores else 0.0
    mean_neg = float(np.mean(neg_scores)) if neg_scores else 0.0
    neg_polarity = float(np.clip(0.7 * max_neg + 0.3 * mean_neg, 0.0, 1.0))

    return {
        "positive_polarity": round(pos_score, 4),
        "negative_polarity": round(neg_polarity, 4),
    }


def calculate_distribution_entropy(probabilities: Dict[str, float]) -> float:
    """
    Computes normalized Shannon entropy of the probability distribution.
    Lower entropy indicates sharp, concentrated emotion; higher entropy indicates dispersed/uncertain state.
    """
    probs = np.array(list(probabilities.values()), dtype=np.float64)
    total = np.sum(probs)
    if total <= 1e-9:
        return 1.0
    
    norm_probs = probs / total
    # Avoid log(0)
    norm_probs = np.clip(norm_probs, 1e-12, 1.0)
    entropy = -np.sum(norm_probs * np.log(norm_probs))
    max_entropy = np.log(len(probs))
    
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def calculate_emotional_intensity(
    dominant_emotion: str,
    dominant_confidence: float,
    positive_polarity: float,
    negative_polarity: float,
    probabilities: Dict[str, float],
) -> float:
    """
    Dynamic Intensity Formula combining:
    1. Dominant Emotion Probability (weight: 0.45)
    2. Polarity Contrast and Magnitude (weight: 0.25)
    3. Category Severity Weight (weight: 0.20)
    4. Distribution Concentration (1 - Entropy) (weight: 0.10)

    Formula:
    Intensity = clamp(0.45 * P_dom + 0.25 * Polarity_Mag + 0.20 * Sev_Weight + 0.10 * Certainty, 0.0, 1.0)
    """
    # 1. Dominant probability
    p_dom = float(np.clip(dominant_confidence, 0.0, 1.0))

    # 2. Polarity magnitude
    polarity_mag = float(max(positive_polarity, negative_polarity))

    # 3. Category severity weight
    dom_key = dominant_emotion.lower()
    sev_weight = float(EMOTION_SEVERITY_WEIGHTS.get(dom_key, 0.70))

    # 4. Distribution certainty (1 - normalized entropy)
    norm_entropy = calculate_distribution_entropy(probabilities)
    certainty = float(1.0 - norm_entropy)

    raw_intensity = (
        0.45 * p_dom +
        0.25 * polarity_mag +
        0.20 * sev_weight +
        0.10 * certainty
    )

    return float(np.clip(round(raw_intensity, 4), 0.0, 1.0))


def determine_severity_level(intensity: float, dominant_emotion: str) -> str:
    """Maps dynamic intensity value and dominant emotion to standard severity tier."""
    if intensity >= 0.85:
        return "Very High" if intensity >= 0.90 else "High"
    elif intensity >= 0.65:
        return "High"
    elif intensity >= 0.35:
        return "Moderate"
    else:
        return "Low"


def detect_mixed_emotions(
    probabilities: Dict[str, float],
    positive_polarity: float,
    negative_polarity: float,
    threshold: float = DEFAULT_MIXED_THRESHOLD,
) -> bool:
    """
    Dynamically identifies mixed emotions if:
    - 2 or more emotions meet or exceed the mixed threshold, OR
    - Both positive and negative polarities are significantly active (>= 0.25).
    """
    qualifying_emotions = [p for p in probabilities.values() if p >= threshold]
    has_multiple_qualifying = len(qualifying_emotions) >= 2
    has_co_occurring_polarities = (positive_polarity >= 0.25) and (negative_polarity >= 0.25)

    return bool(has_multiple_qualifying or has_co_occurring_polarities)


def format_final_emotional_state(
    dominant_emotion: str,
    severity: str,
    intensity: float,
    is_mixed: bool,
    top_emotions: List[Dict[str, Any]],
) -> str:
    """Formats a user-friendly descriptive string of the final emotional state."""
    if is_mixed and len(top_emotions) >= 2:
        top_names = [e["emotion"] for e in top_emotions[:2]]
        return f"{severity} Mixed Emotional State ({' + '.join(top_names)})"
    return f"{severity}-Intensity {dominant_emotion}"


def analyze_emotional_state(
    text: Optional[str],
    model_type: str = "bert",
    mixed_threshold: float = DEFAULT_MIXED_THRESHOLD,
    top_k: int = 3,
) -> EmotionalState:
    """
    Main entry point for Task 1.
    Performs deep emotion inference and calculates the complete emotional state.
    """
    # 1. Basic validation
    if not text or not str(text).strip():
        return EmotionalState(
            dominant_emotion="Neutral",
            dominant_confidence=0.0,
            emotional_intensity=0.0,
            positive_polarity=0.0,
            negative_polarity=0.0,
            mixed_emotion=False,
            emotion_severity="Low",
            final_emotional_state="Invalid / Empty Input",
            is_valid=False,
            error_message="Input text cannot be empty or whitespace.",
        )

    # 2. Model Inference
    raw_pred = analyze_emotion(text, model_type=model_type, threshold=mixed_threshold)
    if not raw_pred.get("is_valid", False):
        return EmotionalState(
            dominant_emotion="Neutral",
            dominant_confidence=0.0,
            is_valid=False,
            error_message=raw_pred.get("error_message", "Inference error"),
        )

    probabilities: Dict[str, float] = raw_pred["probabilities"]
    dominant_emotion = raw_pred["primary_emotion"]
    dominant_confidence = float(raw_pred["primary_confidence"])

    # 3. Top-K Emotions
    sorted_emotions = sorted(
        probabilities.items(), key=lambda item: item[1], reverse=True
    )
    top_emotions = [
        {"emotion": emo, "confidence": round(float(prob), 4)}
        for emo, prob in sorted_emotions[:top_k]
    ]

    # 4. Polarity
    polarity = calculate_polarity(probabilities)
    pos_polarity = polarity["positive_polarity"]
    neg_polarity = polarity["negative_polarity"]

    # 5. Dynamic Intensity & Severity
    intensity = calculate_emotional_intensity(
        dominant_emotion=dominant_emotion,
        dominant_confidence=dominant_confidence,
        positive_polarity=pos_polarity,
        negative_polarity=neg_polarity,
        probabilities=probabilities,
    )
    severity = determine_severity_level(intensity, dominant_emotion)

    # 6. Mixed Emotion Detection
    is_mixed = detect_mixed_emotions(
        probabilities=probabilities,
        positive_polarity=pos_polarity,
        negative_polarity=neg_polarity,
        threshold=mixed_threshold,
    )

    # 7. Final Descriptive State
    final_state = format_final_emotional_state(
        dominant_emotion=dominant_emotion,
        severity=severity,
        intensity=intensity,
        is_mixed=is_mixed,
        top_emotions=top_emotions,
    )

    return EmotionalState(
        dominant_emotion=dominant_emotion,
        dominant_confidence=dominant_confidence,
        top_emotions=top_emotions,
        emotional_intensity=intensity,
        positive_polarity=pos_polarity,
        negative_polarity=neg_polarity,
        mixed_emotion=is_mixed,
        emotion_severity=severity,
        final_emotional_state=final_state,
        probabilities=probabilities,
        raw_text=text,
        model_used=model_type,
        is_valid=True,
    )
