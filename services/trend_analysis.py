"""
Emotional Trend and User State Tracking Service (Task 6).
Analyzes historical emotion records to identify:
1. Emotion Frequency distribution
2. Emotion Intensity trends over configurable time windows
3. Multi-factor Dominant Emotions (using recency-decayed weighted scoring)
4. Positive and Negative polarity trends
5. Repeated emotional patterns (streaks, chronic intensity, transitions, volatility)
6. Structured Current & Recent Emotional State
7. Historical state-activity synergy for dynamic recommendation re-ranking
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from services.config import EMOTIONS, EMOTION_DISPLAY_NAMES


@dataclass
class TrackedUserState:
    """Structured representation of a user's tracked emotional state and historical trends."""
    dominant_emotion: str = "Neutral"
    current_emotion: str = "Neutral"
    current_intensity: float = 0.0
    recent_emotion_distribution: Dict[str, float] = field(default_factory=dict)
    positive_trend: Dict[str, Any] = field(default_factory=dict)
    negative_trend: Dict[str, Any] = field(default_factory=dict)
    repeated_patterns: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.50
    intensity_trends: Dict[str, Any] = field(default_factory=dict)
    frequency_analysis: Dict[str, Any] = field(default_factory=dict)
    total_historical_checkins: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Converts to serializable dictionary."""
        return asdict(self)


def normalize_emotion_name(emotion: Optional[str]) -> str:
    """Normalizes emotion strings to title case (e.g. 'joy' -> 'Joy')."""
    if not emotion:
        return "Joy"
    key = str(emotion).strip().lower()
    return EMOTION_DISPLAY_NAMES.get(key, key.title())


def calculate_emotion_frequency(
    emotion_history: List[Dict[str, Any]],
    window_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    1. Emotion Frequency:
    Counts each detected emotion across all historical records (or within a recent window).
    Supports all 6 core emotions: Joy, Sadness, Anger, Fear, Surprise, Disgust.

    Returns:
        Dict with raw counts, normalized proportions, total entries, and sorted ranking.
    """
    all_emotions = [EMOTION_DISPLAY_NAMES[e] for e in EMOTIONS]
    counts = {emo: 0 for emo in all_emotions}
    
    records = emotion_history if window_size is None else emotion_history[-window_size:]
    valid_entries = 0

    for entry in records:
        dom = entry.get("dominant_emotion")
        if dom:
            norm = normalize_emotion_name(dom)
            if norm in counts:
                counts[norm] += 1
                valid_entries += 1
            else:
                counts[norm] = counts.get(norm, 0) + 1
                valid_entries += 1

    proportions = {
        emo: round(count / valid_entries, 4) if valid_entries > 0 else 0.0
        for emo, count in counts.items()
    }

    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "counts": counts,
        "proportions": proportions,
        "total_entries": valid_entries,
        "ranked_emotions": [{"emotion": k, "count": v, "proportion": proportions[k]} for k, v in ranked],
    }


def calculate_intensity_trends(
    emotion_history: List[Dict[str, Any]],
    window_size: int = 5,
) -> Dict[str, Any]:
    """
    2. Emotion Intensity Over Time:
    Tracks intensity per historical prediction, computes historical baseline, recent moving average,
    intensity delta, trajectory, and volatility across configurable windows.
    """
    if not emotion_history:
        return {
            "current_intensity": 0.0,
            "historical_mean": 0.0,
            "recent_mean": 0.0,
            "delta": 0.0,
            "direction": "stable",
            "volatility": 0.0,
            "min_intensity": 0.0,
            "max_intensity": 0.0,
            "window_size": window_size,
            "series": [],
        }

    intensities = [float(e.get("intensity", 0.0)) for e in emotion_history]
    historical_mean = float(np.mean(intensities)) if intensities else 0.0
    current_intensity = intensities[-1] if intensities else 0.0

    recent_slice = intensities[-window_size:] if len(intensities) >= window_size else intensities
    recent_mean = float(np.mean(recent_slice)) if recent_slice else 0.0

    delta = round(recent_mean - historical_mean, 4)

    # Direction classification based on intensity delta
    if delta >= 0.08:
        direction = "escalating"
    elif delta <= -0.08:
        direction = "decreasing"
    else:
        direction = "stable"

    volatility = float(np.std(intensities)) if len(intensities) > 1 else 0.0

    series = [
        {
            "index": i + 1,
            "timestamp": e.get("timestamp", f"Check-in #{i+1}"),
            "intensity": round(float(e.get("intensity", 0.0)), 4),
            "dominant_emotion": normalize_emotion_name(e.get("dominant_emotion", "Joy")),
        }
        for i, e in enumerate(emotion_history)
    ]

    return {
        "current_intensity": round(current_intensity, 4),
        "historical_mean": round(historical_mean, 4),
        "recent_mean": round(recent_mean, 4),
        "delta": delta,
        "direction": direction,
        "volatility": round(volatility, 4),
        "min_intensity": round(float(min(intensities)), 4) if intensities else 0.0,
        "max_intensity": round(float(max(intensities)), 4) if intensities else 0.0,
        "window_size": window_size,
        "series": series,
    }


def determine_dominant_emotions(
    emotion_history: List[Dict[str, Any]],
    window_size: Optional[int] = None,
    decay_factor: float = 0.90,
) -> Dict[str, Any]:
    """
    3. Dominant Emotions:
    Identifies the most frequent and relevant emotions over time using exponential recency decay.
    Does NOT simply select the latest emotion.
    
    Formula:
    Score(Emotion E) = sum_{t where Emotion == E} (decay_factor^(T - t)) * (0.60 + 0.40 * Intensity_t)
    """
    if not emotion_history:
        return {
            "dominant_emotion": "Joy",
            "dominant_score": 0.0,
            "ranked_emotions": [],
            "is_single_dominant": True,
        }

    records = emotion_history if window_size is None else emotion_history[-window_size:]
    t_total = len(records)
    
    all_emotions = [EMOTION_DISPLAY_NAMES[e] for e in EMOTIONS]
    scores = {emo: 0.0 for emo in all_emotions}
    counts = {emo: 0 for emo in all_emotions}
    intensities = {emo: [] for emo in all_emotions}

    for idx, entry in enumerate(records):
        emo_name = normalize_emotion_name(entry.get("dominant_emotion", "Joy"))
        if emo_name not in scores:
            scores[emo_name] = 0.0
            counts[emo_name] = 0
            intensities[emo_name] = []

        intensity = float(entry.get("intensity", 0.50))
        # Recency weight: latest entry has weight 1.0; older entries decay by decay_factor
        recency_weight = decay_factor ** (t_total - 1 - idx)
        entry_weight = recency_weight * (0.60 + 0.40 * intensity)

        scores[emo_name] += entry_weight
        counts[emo_name] += 1
        intensities[emo_name].append(intensity)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_emotion, top_score = ranked[0] if ranked else ("Joy", 0.0)

    # Check if there is a distinct single leader
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    is_single_dominant = (top_score - second_score) >= (0.25 * max(top_score, 1e-6))

    ranked_details = [
        {
            "emotion": emo,
            "score": round(score, 4),
            "frequency": counts.get(emo, 0),
            "mean_intensity": round(float(np.mean(intensities[emo])), 4) if intensities[emo] else 0.0,
        }
        for emo, score in ranked
    ]

    return {
        "dominant_emotion": top_emotion,
        "dominant_score": round(top_score, 4),
        "ranked_emotions": ranked_details,
        "is_single_dominant": is_single_dominant,
    }


def calculate_polarity_trends(
    emotion_history: List[Dict[str, Any]],
    recent_window: int = 5,
) -> Dict[str, Any]:
    """
    4. Positive / Negative Trends:
    Calculates positive and negative emotional trends from historical data.
    Compares recent period vs previous/baseline period.
    """
    if not emotion_history:
        return {
            "positive_trend": {"direction": "stable", "delta": 0.0, "slope": 0.0, "recent_mean": 0.0},
            "negative_trend": {"direction": "stable", "delta": 0.0, "slope": 0.0, "recent_mean": 0.0},
            "polarity_shift": 0.0,
            "net_polarity_status": "Balanced",
        }

    pos_series = [float(e.get("positive_polarity", 0.0)) for e in emotion_history]
    neg_series = [float(e.get("negative_polarity", 0.0)) for e in emotion_history]
    n = len(emotion_history)

    # Compute baseline vs recent window
    if n <= recent_window:
        # Split in half if fewer items than recent_window
        mid = max(1, n // 2)
        base_pos = pos_series[:mid]
        rec_pos = pos_series[mid:] if mid < n else pos_series
        base_neg = neg_series[:mid]
        rec_neg = neg_series[mid:] if mid < n else neg_series
    else:
        base_pos = pos_series[:-recent_window]
        rec_pos = pos_series[-recent_window:]
        base_neg = neg_series[:-recent_window]
        rec_neg = neg_series[-recent_window:]

    mean_base_pos = float(np.mean(base_pos)) if base_pos else 0.0
    mean_rec_pos = float(np.mean(rec_pos)) if rec_pos else 0.0
    pos_delta = round(mean_rec_pos - mean_base_pos, 4)

    mean_base_neg = float(np.mean(base_neg)) if base_neg else 0.0
    mean_rec_neg = float(np.mean(rec_neg)) if rec_neg else 0.0
    neg_delta = round(mean_rec_neg - mean_base_neg, 4)

    # Simple linear slope across full series
    def compute_slope(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values)
        slope, _ = np.polyfit(x, y, 1)
        return float(round(slope, 4))

    pos_slope = compute_slope(pos_series)
    neg_slope = compute_slope(neg_series)

    # Positive trend direction
    if pos_delta >= 0.06 or pos_slope >= 0.03:
        pos_dir = "improving"
    elif pos_delta <= -0.06 or pos_slope <= -0.03:
        pos_dir = "declining"
    else:
        pos_dir = "stable"

    # Negative trend direction
    if neg_delta >= 0.06 or neg_slope >= 0.03:
        neg_dir = "escalating"
    elif neg_delta <= -0.06 or neg_slope <= -0.03:
        neg_dir = "recovering"
    else:
        neg_dir = "stable"

    # Polarity shift: recent net polarity minus baseline net polarity
    rec_net = mean_rec_pos - mean_rec_neg
    base_net = mean_base_pos - mean_base_neg
    polarity_shift = round(rec_net - base_net, 4)

    if polarity_shift >= 0.10:
        net_status = "Positive Shift"
    elif polarity_shift <= -0.10:
        net_status = "Negative Shift"
    else:
        net_status = "Balanced / Steady"

    return {
        "positive_trend": {
            "direction": pos_dir,
            "delta": pos_delta,
            "slope": pos_slope,
            "recent_mean": round(mean_rec_pos, 4),
            "baseline_mean": round(mean_base_pos, 4),
        },
        "negative_trend": {
            "direction": neg_dir,
            "delta": neg_delta,
            "slope": neg_slope,
            "recent_mean": round(mean_rec_neg, 4),
            "baseline_mean": round(mean_base_neg, 4),
        },
        "polarity_shift": polarity_shift,
        "net_polarity_status": net_status,
    }


def detect_repeated_patterns(
    emotion_history: List[Dict[str, Any]],
    min_streak: int = 2,
    high_intensity_thresh: float = 0.70,
) -> List[Dict[str, Any]]:
    """
    5. Repeated Emotional Patterns:
    Detects:
    - Persistent emotional streaks (e.g. 2+ or 3+ consecutive same emotions).
    - Chronic high-intensity states (multiple consecutive high intensity sessions).
    - Recurring emotion transitions (e.g. Fear -> Sadness or Anger -> Sadness).
    - Emotional volatility (rapid back-and-forth swings between high positive and high negative).
    """
    if len(emotion_history) < min_streak:
        return []

    patterns: List[Dict[str, Any]] = []
    emotions = [normalize_emotion_name(e.get("dominant_emotion", "Joy")) for e in emotion_history]
    intensities = [float(e.get("intensity", 0.0)) for e in emotion_history]
    n = len(emotions)

    # 1. Emotional Streaks (Consecutive identical emotions)
    current_streak = 1
    streak_emotion = emotions[-1]
    for i in range(n - 2, -1, -1):
        if emotions[i] == streak_emotion:
            current_streak += 1
        else:
            break

    if current_streak >= min_streak:
        severity = "High" if streak_emotion in ["Fear", "Sadness", "Anger", "Disgust"] and current_streak >= 3 else "Moderate"
        patterns.append({
            "pattern_type": "emotional_streak",
            "name": f"Persistent {streak_emotion} Streak",
            "description": f"User has logged {current_streak} consecutive check-ins with dominant emotion '{streak_emotion}'.",
            "streak_length": current_streak,
            "emotion": streak_emotion,
            "severity": severity,
            "recommendation_hint": "prioritize_calming_and_grounding" if streak_emotion in ["Fear", "Sadness", "Anger"] else "sustain_positive_momentum",
        })

    # 2. Chronic High Intensity
    recent_k = min(4, n)
    recent_intensities = intensities[-recent_k:]
    high_intensity_count = sum(1 for val in recent_intensities if val >= high_intensity_thresh)

    if high_intensity_count >= 2:
        mean_recent_int = float(np.mean(recent_intensities))
        patterns.append({
            "pattern_type": "chronic_high_intensity",
            "name": "Chronic High Intensity State",
            "description": f"{high_intensity_count} out of the last {recent_k} check-ins showed elevated intensity (Avg: {mean_recent_int:.2f}).",
            "intensity_average": round(mean_recent_int, 4),
            "severity": "High",
            "recommendation_hint": "prioritize_short_somatic_interventions",
        })

    # 3. Recurring Transitions (e.g. Fear -> Sadness or Anger -> Sadness)
    if n >= 3:
        transitions = [(emotions[i], emotions[i + 1]) for i in range(n - 1)]
        trans_counts: Dict[Tuple[str, str], int] = {}
        for t in transitions:
            trans_counts[t] = trans_counts.get(t, 0) + 1

        for (e1, e2), count in trans_counts.items():
            if count >= 2 and e1 != e2 and (e1 in ["Fear", "Anger", "Sadness"] or e2 in ["Fear", "Anger", "Sadness"]):
                patterns.append({
                    "pattern_type": "recurrent_transition",
                    "name": f"Recurring Transition ({e1} → {e2})",
                    "description": f"Pattern of transitioning from '{e1}' to '{e2}' observed {count} times.",
                    "from_emotion": e1,
                    "to_emotion": e2,
                    "frequency": count,
                    "severity": "Moderate",
                    "recommendation_hint": "address_underlying_cycle",
                })

    # 4. Emotional Volatility (Rapid alternating polarity swings)
    if n >= 4:
        pos_series = [float(e.get("positive_polarity", 0.0)) for e in emotion_history[-6:]]
        neg_series = [float(e.get("negative_polarity", 0.0)) for e in emotion_history[-6:]]
        swings = 0
        for i in range(len(pos_series) - 1):
            curr_pos = pos_series[i] > 0.50
            next_neg = neg_series[i + 1] > 0.50
            curr_neg = neg_series[i] > 0.50
            next_pos = pos_series[i + 1] > 0.50
            if (curr_pos and next_neg) or (curr_neg and next_pos):
                swings += 1

        if swings >= 2:
            patterns.append({
                "pattern_type": "emotional_volatility",
                "name": "High Emotional Volatility",
                "description": f"Observed {swings} rapid polarity swings across recent check-ins.",
                "severity": "Moderate",
                "recommendation_hint": "stabilizing_mindfulness_and_pacing",
            })

    return patterns


def build_tracked_user_state(
    current_state: Any,
    emotion_history: List[Dict[str, Any]],
    window_size: int = 5,
) -> TrackedUserState:
    """
    6. Recent Emotional State:
    Generates a structured current-state object containing:
    - dominant_emotion (from historical multi-factor analysis)
    - current_emotion
    - current_intensity
    - recent_emotion_distribution
    - positive_trend
    - negative_trend
    - repeated_patterns
    - confidence
    """
    # Extract current emotion and intensity
    if hasattr(current_state, "dominant_emotion"):
        curr_emotion = normalize_emotion_name(current_state.dominant_emotion)
        curr_intensity = float(getattr(current_state, "emotional_intensity", 0.50))
        curr_conf = float(getattr(current_state, "dominant_confidence", 0.80))
    elif isinstance(current_state, dict):
        curr_emotion = normalize_emotion_name(current_state.get("dominant_emotion", "Joy"))
        curr_intensity = float(current_state.get("emotional_intensity", 0.50))
        curr_conf = float(current_state.get("dominant_confidence", 0.80))
    else:
        curr_emotion = "Joy"
        curr_intensity = 0.50
        curr_conf = 0.80

    # If history is empty, initialize single-entry history with current state
    effective_history = list(emotion_history)
    if not effective_history:
        effective_history = [{
            "dominant_emotion": curr_emotion,
            "intensity": curr_intensity,
            "positive_polarity": 0.80 if curr_emotion == "Joy" else 0.10,
            "negative_polarity": 0.10 if curr_emotion == "Joy" else 0.80,
            "timestamp": datetime.utcnow().isoformat(),
        }]

    # 1. Frequency
    freq_data = calculate_emotion_frequency(effective_history, window_size=window_size)
    recent_dist = freq_data["proportions"]

    # 2. Intensity Trends
    intensity_data = calculate_intensity_trends(effective_history, window_size=window_size)

    # 3. Dominant Emotion Over Time
    dominant_data = determine_dominant_emotions(effective_history, window_size=window_size)
    historical_dominant = dominant_data["dominant_emotion"]

    # 4. Polarity Trends
    polarity_data = calculate_polarity_trends(effective_history, recent_window=window_size)

    # 5. Repeated Patterns
    patterns = detect_repeated_patterns(effective_history)

    # 6. Overall Confidence Calculation
    sample_factor = min(1.0, len(effective_history) / 5.0)
    consistency_factor = 0.85 if dominant_data["is_single_dominant"] else 0.70
    overall_conf = float(np.clip(
        0.50 * curr_conf + 0.30 * sample_factor + 0.20 * consistency_factor,
        0.10,
        1.0,
    ))

    return TrackedUserState(
        dominant_emotion=historical_dominant,
        current_emotion=curr_emotion,
        current_intensity=round(curr_intensity, 4),
        recent_emotion_distribution=recent_dist,
        positive_trend=polarity_data["positive_trend"],
        negative_trend=polarity_data["negative_trend"],
        repeated_patterns=patterns,
        confidence=round(overall_conf, 4),
        intensity_trends=intensity_data,
        frequency_analysis=freq_data,
        total_historical_checkins=len(effective_history),
    )


def calculate_historical_emotion_synergy(
    item_activity: str,
    item_tags: List[str],
    tracked_state: TrackedUserState,
    user_recommendation_history: List[Dict[str, Any]],
) -> float:
    """
    7. Recommendation Integration:
    Evaluates dynamic historical synergy between user's past successful interactions and current emotional patterns.
    
    For example:
    - If relaxation content was repeatedly accepted/liked during high-stress/fear states,
      increase its relevance when a similar state or persistent streak occurs.
    - Dynamically queries past interactions without hardcoding items.
    """
    if not user_recommendation_history:
        return 0.50

    act_key = item_activity.strip().lower()
    tag_set = set(t.strip().lower() for t in item_tags)
    
    curr_emo = tracked_state.current_emotion.lower()
    dom_emo = tracked_state.dominant_emotion.lower()
    has_high_distress_pattern = any(
        p.get("pattern_type") in ["emotional_streak", "chronic_high_intensity"]
        and p.get("severity") == "High"
        for p in tracked_state.repeated_patterns
    )

    matching_likes = 0
    total_relevant_events = 0

    for event in user_recommendation_history:
        event_emo = str(event.get("dominant_emotion", "")).strip().lower()
        is_relevant_context = (
            event_emo in [curr_emo, dom_emo] or
            (has_high_distress_pattern and event_emo in ["fear", "sadness", "anger", "disgust"])
        )

        if is_relevant_context:
            total_relevant_events += 1
            if event.get("was_liked", False) or event.get("was_selected", False):
                ev_act = str(event.get("activity_type", "")).strip().lower()
                if ev_act and ev_act == act_key:
                    matching_likes += 2
                elif any(t in event.get("tags", []) for t in tag_set):
                    matching_likes += 1
            elif event.get("was_disliked", False):
                ev_act = str(event.get("activity_type", "")).strip().lower()
                if ev_act and ev_act == act_key:
                    matching_likes -= 2

    if total_relevant_events == 0:
        return 0.50

    synergy_ratio = matching_likes / max(total_relevant_events, 1)
    final_synergy = float(np.clip(0.50 + 0.25 * synergy_ratio, 0.0, 1.0))
    return round(final_synergy, 4)
