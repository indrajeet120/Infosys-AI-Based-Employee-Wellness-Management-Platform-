"""
Hybrid Recommendation Engine and Dynamic Ranking Model.
Combines rule-based, content-based, user-preference matching, emotion similarity,
semantic embeddings, and historical behavior into dynamically ranked wellness recommendations.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import numpy as np

from services.config import (
    DEFAULT_RANKING_WEIGHTS,
    MEDICAL_DISCLAIMER,
)
from services.intensity import EmotionalState
from services.wellness_data import WellnessContent, get_wellness_repository
from services.user_profile import (
    UserProfile,
    UserProfileManager,
    get_user_profile_manager,
    check_collaborative_filtering_availability,
)
from services.semantic_matcher import get_semantic_matcher
from services.trend_analysis import (
    TrackedUserState,
    build_tracked_user_state,
    calculate_historical_emotion_synergy,
)
from services.feedback_learning import get_feedback_manager
from services.explainability import get_explanation_generator


@dataclass
class RecommendationItem:
    """Represents a dynamically scored and explained wellness recommendation."""
    content_id: str
    title: str
    description: str
    content_type: str
    activity_type: str
    difficulty: str
    duration: str
    duration_minutes: int
    target_emotions: List[str]
    tags: List[str]
    final_score: float
    emotion_relevance: float
    intensity_fit: float
    preference_match: float
    semantic_similarity: float
    historical_preference: float
    novelty_score: float
    source_strategies: List[str]
    reason: str
    disclaimer: str = MEDICAL_DISCLAIMER
    # Task 7 Feedback Learning Signals
    historical_acceptance: float = 0.50
    historical_rejection: float = 0.0
    rating_score: float = 0.50
    diversity_score: float = 1.0
    # Task 8 Explanation Factors
    explanation_factors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["recommendation"] = self.title
        d["score"] = self.final_score
        return d


class HybridRecommendationEngine:
    """Orchestrates candidate generation, multi-strategy merging, and dynamic ranking."""
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or dict(DEFAULT_RANKING_WEIGHTS)
        self.repo = get_wellness_repository()
        self.user_manager = get_user_profile_manager()
        self.feedback_manager = get_feedback_manager()
        self.semantic_matcher = get_semantic_matcher()
        self.explanation_generator = get_explanation_generator()

    def calculate_emotion_relevance(self, item: WellnessContent, emotional_state: EmotionalState) -> float:
        """
        Calculates how well the item's target emotions match the user's detected emotion probabilities.
        """
        item_targets = [e.lower() for e in item.target_emotions]
        probs = emotional_state.probabilities
        
        # Sum model probabilities for matching emotions
        matched_probs = [
            probs.get(e.capitalize(), 0.0)
            for e in item_targets
        ]
        
        # Check if dominant emotion matches
        dom_match = 1.0 if emotional_state.dominant_emotion.lower() in item_targets else 0.0
        
        avg_prob = float(np.mean(matched_probs)) if matched_probs else 0.0
        max_prob = float(max(matched_probs)) if matched_probs else 0.0

        relevance = 0.50 * max_prob + 0.30 * dom_match + 0.20 * avg_prob
        return float(np.clip(round(relevance, 4), 0.0, 1.0))

    def calculate_intensity_fit(self, item: WellnessContent, emotional_state: EmotionalState) -> float:
        """
        Evaluates whether the activity type and duration appropriately match the user's emotional intensity.
        High intensity -> Quick, grounding, breathing, somatic activities.
        Low/Moderate intensity -> Journaling, longer reflections, yoga.
        """
        intensity = emotional_state.emotional_intensity
        act = item.activity_type.lower()
        
        # High intensity demands immediate de-escalation (< 8 mins, breathing, somatic PMR, sensory grounding)
        if intensity >= 0.70:
            if act in ["breathing exercise", "relaxation activity"] or item.duration_minutes <= 8:
                fit = 0.95
            elif act in ["journaling", "motivational content"]:
                fit = 0.65
            else:
                fit = 0.50
        elif intensity >= 0.40:
            # Moderate intensity
            fit = 0.85
        else:
            # Low intensity
            if act in ["positive reflection", "physical activity", "journaling"]:
                fit = 0.90
            else:
                fit = 0.75

        return float(np.clip(round(fit, 4), 0.0, 1.0))

    def calculate_preference_match(self, item: WellnessContent, user_profile: UserProfile) -> float:
        """Calculates match with user content type, activity preferences, and language."""
        type_match = 1.0 if item.content_type in user_profile.preferred_content_types else 0.20
        act_match = 1.0 if item.activity_type in user_profile.preferred_activities else 0.30
        lang_match = 1.0 if item.language.lower() == user_profile.preferred_language.lower() else 0.50

        pref_score = 0.45 * act_match + 0.40 * type_match + 0.15 * lang_match
        return float(np.clip(round(pref_score, 4), 0.0, 1.0))

    def calculate_historical_preference(self, item: WellnessContent, user_profile: UserProfile) -> float:
        """
        Calculates historical preference based on user's past likes, dislikes, and shared tags.
        """
        if item.content_id in user_profile.liked_content:
            return 1.0
        if item.content_id in user_profile.disliked_content:
            return 0.0

        # Tag-level historical synergy
        liked_items = [self.repo.get_by_id(cid) for cid in user_profile.liked_content if self.repo.get_by_id(cid)]
        liked_tags = set(tag for l_item in liked_items for tag in l_item.tags)

        if liked_tags:
            shared_tags = set(item.tags).intersection(liked_tags)
            tag_boost = min(len(shared_tags) / max(len(item.tags), 1), 1.0)
            return float(round(0.50 + 0.50 * tag_boost, 4))

        return 0.50  # Neutral baseline when no history exists

    def calculate_novelty_score(self, item: WellnessContent, user_profile: UserProfile) -> float:
        """Calculates novelty score: unconsumed items get higher score; repeat items decay."""
        selection_count = user_profile.previously_selected_content.count(item.content_id)
        if selection_count == 0:
            return 1.0
        elif selection_count == 1:
            return 0.60
        else:
            return float(max(0.20, 1.0 - 0.35 * selection_count))

    def construct_explanation(
        self,
        item: WellnessContent,
        dominant_emotion: str,
        emotional_intensity: float,
        source_strategies: List[str],
        preference_match: float,
        semantic_sim: float,
        historical_synergy: float = 0.50,
        acceptance_score: float = 0.50,
        rating_score: float = 0.50,
    ) -> str:
        """Constructs dynamic, transparent reason for why this item was recommended."""
        reasons = []
        if acceptance_score >= 0.70:
            reasons.append("matches activities you previously accepted")
        if rating_score >= 0.75:
            reasons.append("highly rated by you")
        if "historical_pattern_synergy" in source_strategies or historical_synergy >= 0.65:
            reasons.append(f"historically effective for your {dominant_emotion.lower()} patterns")
        if "emotion_similarity" in source_strategies:
            reasons.append(f"targeted for {dominant_emotion.lower()}")
        if "semantic_similarity" in source_strategies:
            reasons.append(f"{semantic_sim * 100:.0f}% semantic alignment")
        if "preference_matching" in source_strategies and preference_match > 0.6:
            reasons.append(f"matches your preferred {item.activity_type}")
        if "rule_based" in source_strategies:
            reasons.append(f"calibrated for intensity ({emotional_intensity:.2f})")

        reason_str = ", ".join(reasons) if reasons else f"relevant for {item.activity_type}"
        return f"Recommended because it is {reason_str}."

    def recommend(
        self,
        emotional_state: EmotionalState,
        user_id: str = "default_user",
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full hybrid candidate generation, scoring, and dynamic ranking workflow,
        incorporating historical trend tracking and pattern-aware synergy.
        """
        base_weights = weights or self.weights
        active_weights = dict(base_weights)
        user_profile = self.user_manager.get_or_create_profile(user_id)
        items = self.repo.get_all()

        # Step 0: Build Structured Tracked User State from historical check-ins
        tracked_user_state = build_tracked_user_state(emotional_state, user_profile.emotion_history)
        rejected_ids = self.feedback_manager.get_rejected_content_ids(user_id)

        # Dynamic weight adaptation based on detected patterns
        has_distress_pattern = any(
            p.get("severity") == "High" or p.get("pattern_type") in ["emotional_streak", "chronic_high_intensity"]
            for p in tracked_user_state.repeated_patterns
        )
        if has_distress_pattern:
            # Shift weight towards intensity de-escalation and historical synergy
            active_weights["intensity_weight"] = active_weights.get("intensity_weight", 0.15) + 0.05
            active_weights["history_weight"] = active_weights.get("history_weight", 0.10) + 0.05
            active_weights["preference_weight"] = max(0.10, active_weights.get("preference_weight", 0.20) - 0.05)

        # Step 1: Semantic Matching
        semantic_results = self.semantic_matcher.match_content(
            dominant_emotion=emotional_state.dominant_emotion,
            emotional_intensity=emotional_state.emotional_intensity,
            positive_polarity=emotional_state.positive_polarity,
            negative_polarity=emotional_state.negative_polarity,
            final_emotional_state=emotional_state.final_emotional_state,
            raw_text=emotional_state.raw_text,
        )

        candidates: List[RecommendationItem] = []

        for item in items:
            # Filter out explicitly disliked content immediately
            if item.content_id in user_profile.disliked_content or item.content_id in rejected_ids:
                continue

            # Candidate Source Strategies Tracking
            source_strategies = []

            # 1. Emotion Relevance
            e_rel = self.calculate_emotion_relevance(item, emotional_state)
            if e_rel >= 0.35:
                source_strategies.append("emotion_similarity")

            # 2. Intensity Fit & Rule-Based
            i_fit = self.calculate_intensity_fit(item, emotional_state)
            if i_fit >= 0.70:
                source_strategies.append("rule_based")

            # 3. Preference Match
            p_match = self.calculate_preference_match(item, user_profile)
            if p_match >= 0.60:
                source_strategies.append("preference_matching")

            # 4. Semantic Similarity
            sem_data = semantic_results.get(item.content_id, {})
            sem_sim = float(sem_data.get("semantic_similarity", 0.50))
            if sem_sim >= 0.55:
                source_strategies.append("semantic_similarity")

            # 5. Historical Preference & Emotional Synergy
            h_score_base = self.calculate_historical_preference(item, user_profile)
            historical_synergy = calculate_historical_emotion_synergy(
                item_activity=item.activity_type,
                item_tags=item.tags,
                tracked_state=tracked_user_state,
                user_recommendation_history=user_profile.recommendation_history,
            )
            # Combine direct tag/item historical preference with state-activity synergy
            h_score = float(np.clip(round(0.55 * h_score_base + 0.45 * historical_synergy, 4), 0.0, 1.0))

            if h_score > 0.50 or historical_synergy >= 0.65:
                source_strategies.append("historical_preference")
            if historical_synergy >= 0.65:
                source_strategies.append("historical_pattern_synergy")

            # 6. Novelty
            n_score = self.calculate_novelty_score(item, user_profile)

            # 7. Task 7 Feedback Learning Signals
            hist_accept = self.feedback_manager.calculate_historical_acceptance_score(
                content_id=item.content_id,
                activity_type=item.activity_type,
                tags=item.tags,
                user_id=user_id,
                current_emotion=emotional_state.dominant_emotion,
            )
            hist_reject = self.feedback_manager.calculate_historical_rejection_penalty(
                content_id=item.content_id,
                activity_type=item.activity_type,
                tags=item.tags,
                user_id=user_id,
                current_emotion=emotional_state.dominant_emotion,
            )
            rating_score = self.feedback_manager.calculate_item_rating_score(
                content_id=item.content_id,
                activity_type=item.activity_type,
                user_id=user_id,
            )

            if hist_accept >= 0.70:
                source_strategies.append("feedback_acceptance")
            if rating_score >= 0.75:
                source_strategies.append("high_user_rating")

            # Penalties
            duplicate_pen = 0.0
            if item.content_id in user_profile.previously_selected_content:
                duplicate_pen = active_weights.get("duplicate_penalty", 0.20) * 0.5

            low_rel_pen = 0.0
            if e_rel < 0.20 and sem_sim < 0.45:
                low_rel_pen = active_weights.get("low_relevance_penalty", 0.30)

            rejection_pen = active_weights.get("rejection_penalty_weight", 0.25) * hist_reject

            # Dynamic Ranking Formula with Task 7 Feedback Weighting
            w_emo = active_weights.get("emotion_weight", 0.25)
            w_int = active_weights.get("intensity_weight", 0.15)
            w_pref = active_weights.get("preference_weight", 0.15)
            w_sim = active_weights.get("similarity_weight", 0.15)
            w_hist = active_weights.get("history_weight", 0.10)
            w_accept = active_weights.get("acceptance_weight", 0.10)
            w_rate = active_weights.get("rating_weight", 0.05)
            w_nov = active_weights.get("novelty_weight", 0.05)

            final_score = (
                w_emo * e_rel +
                w_int * i_fit +
                w_pref * p_match +
                w_sim * sem_sim +
                w_hist * h_score +
                w_accept * hist_accept +
                w_rate * rating_score +
                w_nov * n_score -
                duplicate_pen -
                low_rel_pen -
                rejection_pen
            )
            final_score = float(np.clip(round(final_score, 4), 0.0, 1.0))

            # Discard very low relevance candidates
            if final_score < 0.15:
                continue

            # Generate dynamic evidence-backed explanation and factor breakdown
            explanation = self.explanation_generator.generate_explanation(
                item=item,
                emotional_state=emotional_state,
                user_profile=user_profile,
                tracked_user_state=tracked_user_state,
                e_rel=e_rel,
                i_fit=i_fit,
                p_match=p_match,
                sem_sim=sem_sim,
                h_score=h_score,
                hist_accept=hist_accept,
                hist_reject=hist_reject,
                rating_score=rating_score,
                active_weights=active_weights,
            )
            explanation_factors_dicts = [f.to_dict() for f in explanation.explanation_factors]
            reason = explanation.reason

            rec_item = RecommendationItem(
                content_id=item.content_id,
                title=item.title,
                description=item.description,
                content_type=item.content_type,
                activity_type=item.activity_type,
                difficulty=item.difficulty,
                duration=item.duration,
                duration_minutes=item.duration_minutes,
                target_emotions=item.target_emotions,
                tags=item.tags,
                final_score=final_score,
                emotion_relevance=e_rel,
                intensity_fit=i_fit,
                preference_match=p_match,
                semantic_similarity=sem_sim,
                historical_preference=h_score,
                novelty_score=n_score,
                source_strategies=source_strategies,
                reason=reason,
                historical_acceptance=hist_accept,
                historical_rejection=hist_reject,
                rating_score=rating_score,
                diversity_score=1.0,
                explanation_factors=explanation_factors_dicts,
            )
            candidates.append(rec_item)

        # Dynamic Diversity-Aware Re-ranking strictly descending
        selected_activities: List[str] = []
        top_recommendations: List[RecommendationItem] = []

        # Sort initial pool by final_score
        sorted_pool = sorted(candidates, key=lambda x: x.final_score, reverse=True)

        for cand in sorted_pool:
            if len(top_recommendations) >= top_k:
                break
            div_score = self.feedback_manager.calculate_diversity_score(
                cand.activity_type,
                selected_activities,
            )
            cand.diversity_score = div_score
            top_recommendations.append(cand)
            selected_activities.append(cand.activity_type.strip().lower())

        # Record recommendation view events in feedback manager and profile
        for rec in top_recommendations:
            self.feedback_manager.record_view(
                user_id=user_id,
                recommendation_id=rec.content_id,
                recommendation_type=rec.activity_type,
                emotion=emotional_state.dominant_emotion,
                intensity=emotional_state.emotional_intensity,
                metadata={"title": rec.title, "tags": rec.tags, "final_score": rec.final_score},
            )
            self.user_manager.record_feedback(
                user_id=user_id,
                content_id=rec.content_id,
                feedback_type="viewed",
                recommendation_score=rec.final_score,
                dominant_emotion=emotional_state.dominant_emotion,
                intensity=emotional_state.emotional_intensity,
                activity_type=rec.activity_type,
                tags=rec.tags,
            )

        collab_status = check_collaborative_filtering_availability()

        return {
            "user_id": user_id,
            "emotional_state": emotional_state.to_dict(),
            "tracked_user_state": tracked_user_state.to_dict(),
            "recommendations": [r.to_dict() for r in top_recommendations],
            "total_candidates_evaluated": len(items),
            "qualified_candidates": len(candidates),
            "collaborative_filtering_status": collab_status,
            "ranking_weights_used": active_weights,
            "disclaimer": MEDICAL_DISCLAIMER,
        }


# Singleton engine
_ENGINE_INSTANCE: Optional[HybridRecommendationEngine] = None


def get_recommendation_engine() -> HybridRecommendationEngine:
    """Returns singleton instance of HybridRecommendationEngine."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = HybridRecommendationEngine()
    return _ENGINE_INSTANCE


def get_personalized_recommendations(
    text: str,
    user_id: str = "default_user",
    model_type: str = "bert",
    top_k: int = 5,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Convenience end-to-end entry point:
    Text -> Emotional State -> User Emotion History Recording -> Tracked State -> Hybrid Recommender -> Ranked Recommendations.
    """
    from services.intensity import analyze_emotional_state

    # 1. Analyze emotional state
    emotional_state = analyze_emotional_state(text, model_type=model_type)
    if not emotional_state.is_valid:
        return {
            "is_valid": False,
            "error_message": emotional_state.error_message,
            "recommendations": [],
        }

    # 2. Record emotional entry in profile
    user_manager = get_user_profile_manager()
    user_manager.record_emotion_entry(
        user_id=user_id,
        dominant_emotion=emotional_state.dominant_emotion,
        intensity=emotional_state.emotional_intensity,
        positive_polarity=emotional_state.positive_polarity,
        negative_polarity=emotional_state.negative_polarity,
        final_emotional_state=emotional_state.final_emotional_state,
        raw_text=text,
        probabilities=emotional_state.probabilities,
        confidence=emotional_state.dominant_confidence,
        model_used=model_type,
    )

    # 3. Generate recommendations
    engine = get_recommendation_engine()
    result = engine.recommend(
        emotional_state=emotional_state,
        user_id=user_id,
        top_k=top_k,
        weights=weights,
    )
    result["is_valid"] = True
    return result
