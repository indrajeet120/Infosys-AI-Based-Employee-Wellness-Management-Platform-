"""
Recommendation Explainability Service for AI-Based Employee Wellness Management Platform.

Calculates evidence across 6 recommendation factors:
1. Detected emotion
2. Emotion intensity
3. User preferences
4. Historical behavior
5. Semantic/content relevance
6. Previous recommendation feedback

Generates dynamic, evidence-grounded explanations and structured factor breakdowns
without generic hardcoded responses or medical/clinical claims.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import numpy as np

from services.intensity import EmotionalState
from services.wellness_data import WellnessContent
from services.user_profile import UserProfile
from services.trend_analysis import TrackedUserState


@dataclass
class ExplanationFactor:
    """Represents evidence breakdown for a single recommendation ranking factor."""
    factor_type: str  # "emotion", "intensity", "preference", "historical_behavior", "content_relevance", "previous_feedback"
    name: str
    score: float  # Normalized 0.0 to 1.0
    weighted_contribution: float
    detail: str
    is_strong: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_type": self.factor_type,
            "name": self.name,
            "score": float(np.clip(round(self.score, 4), 0.0, 1.0)),
            "weighted_contribution": float(round(self.weighted_contribution, 4)),
            "detail": self.detail,
            "is_strong": bool(self.is_strong),
        }


@dataclass
class RecommendationExplanation:
    """Encapsulates the dynamic reason and factor breakdowns for a recommendation."""
    recommendation: str
    score: float
    reason: str
    explanation_factors: List[ExplanationFactor] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "score": float(np.clip(round(self.score, 4), 0.0, 1.0)),
            "reason": self.reason,
            "explanation_factors": [
                f.to_dict() if hasattr(f, "to_dict") else f
                for f in self.explanation_factors
            ],
        }


class ExplanationGenerator:
    """Generates evidence-backed explanation factors and dynamic natural-language reasons."""

    def calculate_emotion_factor(
        self,
        item: WellnessContent,
        emotional_state: EmotionalState,
        e_rel: float,
        w_emo: float,
    ) -> ExplanationFactor:
        """Calculates evidence for detected emotion matching."""
        score = float(np.clip(round(e_rel, 4), 0.0, 1.0))
        weighted_contrib = float(round(w_emo * score, 4))
        dom_emo = emotional_state.dominant_emotion.capitalize()
        item_targets = [e.capitalize() for e in item.target_emotions]

        if dom_emo in item_targets:
            detail = f"Targeted for your detected emotion of {dom_emo} (emotion relevance: {score * 100:.0f}%)."
            is_strong = score >= 0.50
        elif any(probs > 0.20 for emo, probs in emotional_state.probabilities.items() if emo.capitalize() in item_targets):
            matched = [emo for emo, p in emotional_state.probabilities.items() if p > 0.20 and emo.capitalize() in item_targets]
            detail = f"Aligns with secondary emotional signals ({', '.join(matched)})."
            is_strong = score >= 0.40
        else:
            detail = f"General wellness relevance for {dom_emo} emotional context."
            is_strong = False

        return ExplanationFactor(
            factor_type="emotion",
            name="Emotion Factor",
            score=score,
            weighted_contribution=weighted_contrib,
            detail=detail,
            is_strong=is_strong,
        )

    def calculate_intensity_factor(
        self,
        item: WellnessContent,
        emotional_state: EmotionalState,
        i_fit: float,
        w_int: float,
    ) -> ExplanationFactor:
        """Calculates evidence for emotional intensity calibration."""
        score = float(np.clip(round(i_fit, 4), 0.0, 1.0))
        weighted_contrib = float(round(w_int * score, 4))
        intensity = emotional_state.emotional_intensity
        dur = item.duration or f"{item.duration_minutes} min"
        act_type = item.activity_type

        if intensity >= 0.70:
            if item.duration_minutes <= 8 or act_type.lower() in ["breathing exercise", "relaxation activity"]:
                detail = f"Calibrated for high emotional intensity ({intensity:.2f}) with a short {dur} {act_type} for immediate grounding."
                is_strong = True
            else:
                detail = f"Suitable for high emotional intensity ({intensity:.2f}) de-escalation."
                is_strong = score >= 0.60
        elif intensity >= 0.40:
            detail = f"Balanced {dur} activity format calibrated for moderate emotional intensity ({intensity:.2f})."
            is_strong = score >= 0.70
        else:
            detail = f"Ideal for low emotional intensity ({intensity:.2f}) reflective engagement."
            is_strong = score >= 0.75

        return ExplanationFactor(
            factor_type="intensity",
            name="Intensity Factor",
            score=score,
            weighted_contribution=weighted_contrib,
            detail=detail,
            is_strong=is_strong,
        )

    def calculate_preference_factor(
        self,
        item: WellnessContent,
        user_profile: UserProfile,
        p_match: float,
        w_pref: float,
    ) -> ExplanationFactor:
        """Calculates evidence for user explicit activity and content preferences."""
        score = float(np.clip(round(p_match, 4), 0.0, 1.0))
        weighted_contrib = float(round(w_pref * score, 4))
        
        matches = []
        if item.activity_type in user_profile.preferred_activities:
            matches.append(f"preferred activity '{item.activity_type}'")
        if item.content_type in user_profile.preferred_content_types:
            matches.append(f"preferred format '{item.content_type}'")
        if item.language.lower() == user_profile.preferred_language.lower():
            matches.append(f"language '{item.language}'")

        if matches:
            detail = f"Matches your explicit preferences: {', '.join(matches)}."
            is_strong = score >= 0.60
        else:
            detail = "Neutral match with general user content preferences."
            is_strong = False

        return ExplanationFactor(
            factor_type="preference",
            name="Preference Factor",
            score=score,
            weighted_contribution=weighted_contrib,
            detail=detail,
            is_strong=is_strong,
        )

    def calculate_historical_behavior_factor(
        self,
        item: WellnessContent,
        user_profile: UserProfile,
        tracked_user_state: Optional[TrackedUserState],
        h_score: float,
        w_hist: float,
    ) -> ExplanationFactor:
        """Calculates evidence for historical selections, liked tags, and state synergy."""
        score = float(np.clip(round(h_score, 4), 0.0, 1.0))
        weighted_contrib = float(round(w_hist * score, 4))
        
        reasons = []
        if item.content_id in user_profile.liked_content:
            reasons.append("previously marked as liked by you")
        
        select_cnt = user_profile.previously_selected_content.count(item.content_id)
        if select_cnt > 0:
            reasons.append(f"selected by you {select_cnt} time(s) in past check-ins")

        if tracked_user_state:
            dom_emo = tracked_user_state.dominant_emotion.lower()
            if score >= 0.65:
                reasons.append(f"historically effective during your {dom_emo} patterns")

        if reasons:
            detail = f"Historical synergy: {'; '.join(reasons)}."
            is_strong = True
        elif score > 0.50:
            detail = "Shares topic tags with your previously consumed wellness content."
            is_strong = score >= 0.65
        else:
            detail = "No prior historical interaction recorded for this specific activity."
            is_strong = False

        return ExplanationFactor(
            factor_type="historical_behavior",
            name="Historical Behavior Factor",
            score=score,
            weighted_contribution=weighted_contrib,
            detail=detail,
            is_strong=is_strong,
        )

    def calculate_content_relevance_factor(
        self,
        item: WellnessContent,
        sem_sim: float,
        w_sim: float,
    ) -> ExplanationFactor:
        """Calculates evidence for semantic text alignment."""
        score = float(np.clip(round(sem_sim, 4), 0.0, 1.0))
        weighted_contrib = float(round(w_sim * score, 4))

        if score >= 0.70:
            detail = f"Strong semantic alignment ({score * 100:.0f}%) between check-in text and activity content."
            is_strong = True
        elif score >= 0.50:
            detail = f"Moderate semantic content relevance ({score * 100:.0f}%) for your topic."
            is_strong = False
        else:
            detail = f"Baseline semantic relevance ({score * 100:.0f}%)."
            is_strong = False

        return ExplanationFactor(
            factor_type="content_relevance",
            name="Content Relevance Factor",
            score=score,
            weighted_contribution=weighted_contrib,
            detail=detail,
            is_strong=is_strong,
        )

    def calculate_previous_feedback_factor(
        self,
        item: WellnessContent,
        hist_accept: float,
        hist_reject: float,
        rating_score: float,
        w_accept: float,
        w_rate: float,
        w_rej: float,
    ) -> ExplanationFactor:
        """Calculates evidence from explicit user feedback (acceptances, ratings, rejections)."""
        combined_score = float(np.clip(round(0.45 * hist_accept + 0.45 * rating_score - 0.50 * hist_reject, 4), 0.0, 1.0))
        weighted_contrib = float(round(w_accept * hist_accept + w_rate * rating_score - w_rej * hist_reject, 4))

        details = []
        if hist_accept >= 0.70:
            details.append("high historical acceptance rate for this activity type")
        if rating_score >= 0.75:
            details.append("positively rated in your previous interactions")
        if hist_reject > 0.0:
            details.append("underwent minor penalty from past rejection history")

        if details:
            detail = f"Feedback learning: {'; '.join(details)}."
            is_strong = hist_accept >= 0.70 or rating_score >= 0.75
        else:
            detail = "New recommendation with neutral feedback baseline."
            is_strong = False

        return ExplanationFactor(
            factor_type="previous_feedback",
            name="Previous Recommendation Feedback Factor",
            score=combined_score,
            weighted_contribution=weighted_contrib,
            detail=detail,
            is_strong=is_strong,
        )

    def generate_explanation(
        self,
        item: WellnessContent,
        emotional_state: EmotionalState,
        user_profile: UserProfile,
        tracked_user_state: Optional[TrackedUserState],
        e_rel: float,
        i_fit: float,
        p_match: float,
        sem_sim: float,
        h_score: float,
        hist_accept: float,
        hist_reject: float,
        rating_score: float,
        active_weights: Dict[str, float],
    ) -> RecommendationExplanation:
        """
        Orchestrates full factor evidence calculation and dynamic reason synthesis.
        """
        w_emo = active_weights.get("emotion_weight", 0.25)
        w_int = active_weights.get("intensity_weight", 0.15)
        w_pref = active_weights.get("preference_weight", 0.15)
        w_sim = active_weights.get("similarity_weight", 0.15)
        w_hist = active_weights.get("history_weight", 0.10)
        w_accept = active_weights.get("acceptance_weight", 0.10)
        w_rate = active_weights.get("rating_weight", 0.05)
        w_rej = active_weights.get("rejection_penalty_weight", 0.25)

        # 1. Calculate the 6 factors
        f_emo = self.calculate_emotion_factor(item, emotional_state, e_rel, w_emo)
        f_int = self.calculate_intensity_factor(item, emotional_state, i_fit, w_int)
        f_pref = self.calculate_preference_factor(item, user_profile, p_match, w_pref)
        f_hist = self.calculate_historical_behavior_factor(item, user_profile, tracked_user_state, h_score, w_hist)
        f_sim = self.calculate_content_relevance_factor(item, sem_sim, w_sim)
        f_fb = self.calculate_previous_feedback_factor(item, hist_accept, hist_reject, rating_score, w_accept, w_rate, w_rej)

        factors = [f_emo, f_int, f_pref, f_hist, f_sim, f_fb]

        # 2. Dynamic Reason Builder using strongest contributing factors
        phrases = []
        dom_emo = emotional_state.dominant_emotion.lower()
        intensity = emotional_state.emotional_intensity

        # Check Intensity contribution
        if f_int.score >= 0.70 and intensity >= 0.70:
            phrases.append(f"{dom_emo} intensity is relatively high ({intensity:.2f})")

        # Check Preference contribution
        if f_pref.score >= 0.60:
            if item.activity_type in user_profile.preferred_activities:
                phrases.append(f"this type of activity matches your {item.activity_type.lower()} preference")
            elif item.content_type in user_profile.preferred_content_types:
                phrases.append(f"it aligns with your preferred {item.content_type.lower()} format")

        # Check Feedback contribution
        if f_fb.score >= 0.65 or hist_accept >= 0.70 or rating_score >= 0.75:
            phrases.append("similar content received positive feedback previously")

        # Check Historical Behavior contribution
        if f_hist.score >= 0.65 and "historical synergy" in f_hist.detail.lower():
            phrases.append(f"it has proven historically effective for your {dom_emo} patterns")

        # Check Emotion match contribution if not already added by intensity
        if f_emo.score >= 0.50 and not any("intensity" in p for p in phrases):
            phrases.append(f"this activity directly targets your current {dom_emo} state")

        # Check Semantic Relevance contribution
        if f_sim.score >= 0.65 and len(phrases) < 3:
            phrases.append(f"it shares strong semantic relevance ({f_sim.score * 100:.0f}%) with your check-in")

        # Fallback if few phrases selected
        if not phrases:
            if f_emo.score >= 0.35:
                phrases.append(f"it aligns with your {dom_emo} emotional state")
            if f_pref.score >= 0.35:
                phrases.append(f"it fits your {item.activity_type.lower()} activity preference")
            if not phrases:
                phrases.append("it matches your current wellness goals")

        # Deduplicate and pick top 3
        unique_phrases = []
        for p in phrases:
            if p not in unique_phrases:
                unique_phrases.append(p)
        selected_phrases = unique_phrases[:3]

        if len(selected_phrases) == 1:
            reason_str = f"Recommended because {selected_phrases[0]}."
        elif len(selected_phrases) == 2:
            reason_str = f"Recommended because {selected_phrases[0]}, and {selected_phrases[1]}."
        else:
            reason_str = f"Recommended because {selected_phrases[0]}, {selected_phrases[1]}, and {selected_phrases[2]}."

        return RecommendationExplanation(
            recommendation=item.title,
            score=0.0,
            reason=reason_str,
            explanation_factors=factors,
        )


# Singleton instance
_GENERATOR_INSTANCE: Optional[ExplanationGenerator] = None


def get_explanation_generator() -> ExplanationGenerator:
    """Returns singleton instance of ExplanationGenerator."""
    global _GENERATOR_INSTANCE
    if _GENERATOR_INSTANCE is None:
        _GENERATOR_INSTANCE = ExplanationGenerator()
    return _GENERATOR_INSTANCE
