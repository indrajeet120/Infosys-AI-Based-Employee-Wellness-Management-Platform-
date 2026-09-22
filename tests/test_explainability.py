"""
Unit & Integration Tests for Task 8 – Recommendation Explainability.
"""

import pytest
from services.intensity import analyze_emotional_state, EmotionalState
from services.hybrid_recommender import (
    HybridRecommendationEngine,
    get_recommendation_engine,
    get_personalized_recommendations,
)
from services.explainability import (
    ExplanationFactor,
    RecommendationExplanation,
    ExplanationGenerator,
    get_explanation_generator,
)
from services.wellness_data import WellnessContent, get_wellness_repository
from services.user_profile import UserProfileManager
from services.feedback_learning import FeedbackManager


@pytest.fixture
def clean_explainability_environment(tmp_path):
    p_path = tmp_path / "test_exp_profiles.json"
    i_path = tmp_path / "test_exp_interactions.json"
    f_path = tmp_path / "test_exp_feedback.json"
    user_mgr = UserProfileManager(profiles_path=p_path, interactions_path=i_path)
    fb_mgr = FeedbackManager(storage_path=f_path)
    engine = HybridRecommendationEngine()
    engine.user_manager = user_mgr
    engine.feedback_manager = fb_mgr
    return engine, user_mgr, fb_mgr


class TestExplainability:
    def test_baseline_explanation_generation_and_structure(self, clean_explainability_environment):
        engine, user_mgr, fb_mgr = clean_explainability_environment
        state = analyze_emotional_state("I am feeling overwhelmed with exam pressure and panic.", model_type="distilbert")
        
        result = engine.recommend(emotional_state=state, user_id="exp_user_1", top_k=3)
        recs = result["recommendations"]
        assert len(recs) > 0

        for r in recs:
            # Check required structure and backward compatibility
            assert "recommendation" in r
            assert "score" in r
            assert "reason" in r
            assert "explanation_factors" in r
            assert "content_id" in r
            assert "title" in r
            assert "final_score" in r

            assert isinstance(r["recommendation"], str)
            assert isinstance(r["score"], float)
            assert isinstance(r["reason"], str)
            assert len(r["reason"]) > 0
            assert isinstance(r["explanation_factors"], list)
            assert len(r["explanation_factors"]) == 6

            # Verify factor structure
            factor_types = [f["factor_type"] for f in r["explanation_factors"]]
            assert set(factor_types) == {
                "emotion",
                "intensity",
                "preference",
                "historical_behavior",
                "content_relevance",
                "previous_feedback",
            }

            for f in r["explanation_factors"]:
                assert "factor_type" in f
                assert "name" in f
                assert "score" in f
                assert "weighted_contribution" in f
                assert "detail" in f
                assert "is_strong" in f
                assert isinstance(f["score"], float)
                assert 0.0 <= f["score"] <= 1.0
                assert isinstance(f["weighted_contribution"], float)
                assert isinstance(f["is_strong"], bool)
                assert len(f["detail"]) > 0

    def test_explanation_changes_when_emotion_changes(self, clean_explainability_environment):
        engine, _, _ = clean_explainability_environment
        state_fear = analyze_emotional_state("I am terrified of failing my presentation.", model_type="distilbert")
        state_joy = analyze_emotional_state("I am so thrilled and joyful today!", model_type="distilbert")

        res_fear = engine.recommend(emotional_state=state_fear, user_id="exp_user_emotion", top_k=1)
        res_joy = engine.recommend(emotional_state=state_joy, user_id="exp_user_emotion", top_k=1)

        rec_fear = res_fear["recommendations"][0]
        rec_joy = res_joy["recommendations"][0]

        # The reason text must dynamically reflect the detected emotion
        assert "fear" in rec_fear["reason"].lower() or "fear" in rec_fear["explanation_factors"][0]["detail"].lower()
        assert "joy" in rec_joy["reason"].lower() or "joy" in rec_joy["explanation_factors"][0]["detail"].lower()
        assert rec_fear["reason"] != rec_joy["reason"]

    def test_explanation_changes_when_intensity_changes(self, clean_explainability_environment):
        gen = get_explanation_generator()
        repo = get_wellness_repository()
        item = repo.get_by_id("well_001")  # 5-minute deep breathing

        # High intensity state
        high_state = EmotionalState(
            dominant_emotion="Fear",
            emotional_intensity=0.85,
            positive_polarity=0.15,
            negative_polarity=0.85,
            final_emotional_state="distressed",
            raw_text="Panic attack",
            probabilities={"Fear": 0.85, "Joy": 0.02},
            dominant_confidence=0.85,
        )

        # Low intensity state
        low_state = EmotionalState(
            dominant_emotion="Fear",
            emotional_intensity=0.25,
            positive_polarity=0.75,
            negative_polarity=0.25,
            final_emotional_state="mild",
            raw_text="Slight nervousness",
            probabilities={"Fear": 0.25, "Joy": 0.40},
            dominant_confidence=0.25,
        )

        dummy_weights = {"emotion_weight": 0.25, "intensity_weight": 0.15, "preference_weight": 0.15}
        user_mgr = clean_explainability_environment[1]
        prof = user_mgr.get_or_create_profile("intensity_user")

        exp_high = gen.generate_explanation(
            item=item,
            emotional_state=high_state,
            user_profile=prof,
            tracked_user_state=None,
            e_rel=0.8,
            i_fit=0.95,
            p_match=0.5,
            sem_sim=0.5,
            h_score=0.5,
            hist_accept=0.5,
            hist_reject=0.0,
            rating_score=0.5,
            active_weights=dummy_weights,
        )

        exp_low = gen.generate_explanation(
            item=item,
            emotional_state=low_state,
            user_profile=prof,
            tracked_user_state=None,
            e_rel=0.5,
            i_fit=0.75,
            p_match=0.5,
            sem_sim=0.5,
            h_score=0.5,
            hist_accept=0.5,
            hist_reject=0.0,
            rating_score=0.5,
            active_weights=dummy_weights,
        )

        assert "high" in exp_high.reason.lower() or "0.85" in exp_high.reason
        assert exp_high.reason != exp_low.reason

    def test_explanation_changes_when_preferences_change(self, clean_explainability_environment):
        engine, user_mgr, _ = clean_explainability_environment
        
        # User 1 prefers Breathing Exercise
        user_mgr.update_preferences(
            user_id="pref_change_user",
            preferred_activities=["Breathing Exercise"],
            preferred_content_types=["audio"],
        )
        state = analyze_emotional_state("I feel stressed out.", model_type="distilbert")
        res_before = engine.recommend(emotional_state=state, user_id="pref_change_user", top_k=3)
        reason_before = res_before["recommendations"][0]["reason"]

        # User updates preference to Journaling
        user_mgr.update_preferences(
            user_id="pref_change_user",
            preferred_activities=["Journaling"],
            preferred_content_types=["text"],
        )
        res_after = engine.recommend(emotional_state=state, user_id="pref_change_user", top_k=3)
        reason_after = res_after["recommendations"][0]["reason"]

        assert reason_before != reason_after or "journaling" in reason_after.lower()

    def test_explanation_changes_when_feedback_changes(self, clean_explainability_environment):
        engine, _, fb_mgr = clean_explainability_environment
        state = analyze_emotional_state("I am feeling anxious.", model_type="distilbert")

        res_initial = engine.recommend(emotional_state=state, user_id="fb_user", top_k=1)
        item_id = res_initial["recommendations"][0]["content_id"]
        reason_initial = res_initial["recommendations"][0]["reason"]

        # Record positive acceptance feedback
        fb_mgr.record_acceptance(
            user_id="fb_user",
            recommendation_id=item_id,
            recommendation_type="Breathing Exercise",
            emotion="Fear",
            intensity=0.7,
            rating=5.0,
        )

        res_post_fb = engine.recommend(emotional_state=state, user_id="fb_user", top_k=1)
        reason_post_fb = res_post_fb["recommendations"][0]["reason"]
        fb_factor = [f for f in res_post_fb["recommendations"][0]["explanation_factors"] if f["factor_type"] == "previous_feedback"][0]

        assert "positive feedback" in reason_post_fb.lower() or "accepted" in fb_factor["detail"].lower()
        assert fb_factor["is_strong"] is True

    def test_explanation_changes_when_semantic_similarity_changes(self, clean_explainability_environment):
        gen = get_explanation_generator()
        repo = get_wellness_repository()
        item = repo.get_by_id("well_001")
        state = analyze_emotional_state("Feeling anxious", model_type="distilbert")
        prof = clean_explainability_environment[1].get_or_create_profile("sem_user")
        dummy_weights = {"emotion_weight": 0.25, "similarity_weight": 0.25}

        exp_high_sem = gen.generate_explanation(
            item=item,
            emotional_state=state,
            user_profile=prof,
            tracked_user_state=None,
            e_rel=0.5,
            i_fit=0.5,
            p_match=0.3,
            sem_sim=0.92,
            h_score=0.5,
            hist_accept=0.5,
            hist_reject=0.0,
            rating_score=0.5,
            active_weights=dummy_weights,
        )

        exp_low_sem = gen.generate_explanation(
            item=item,
            emotional_state=state,
            user_profile=prof,
            tracked_user_state=None,
            e_rel=0.5,
            i_fit=0.5,
            p_match=0.3,
            sem_sim=0.10,
            h_score=0.5,
            hist_accept=0.5,
            hist_reject=0.0,
            rating_score=0.5,
            active_weights=dummy_weights,
        )

        exp_high_dict = exp_high_sem.to_dict()
        exp_low_dict = exp_low_sem.to_dict()

        sim_factor_high = [f for f in exp_high_dict["explanation_factors"] if f["factor_type"] == "content_relevance"][0]
        sim_factor_low = [f for f in exp_low_dict["explanation_factors"] if f["factor_type"] == "content_relevance"][0]

        assert sim_factor_high["score"] == 0.92
        assert sim_factor_low["score"] == 0.10
        assert sim_factor_high["is_strong"] is True
        assert sim_factor_low["is_strong"] is False

    def test_no_generic_hardcoded_explanation(self, clean_explainability_environment):
        engine, _, _ = clean_explainability_environment
        inputs = [
            "I feel extremely angry and outraged!",
            "I am feeling deep sadness and hopelessness.",
            "I feel surprised and delighted by the good news!",
        ]
        reasons = set()
        for txt in inputs:
            state = analyze_emotional_state(txt, model_type="distilbert")
            res = engine.recommend(emotional_state=state, user_id=f"user_{txt[:5]}", top_k=1)
            reasons.add(res["recommendations"][0]["reason"])

        # Reasons must vary dynamically across distinct inputs
        assert len(reasons) > 1

    def test_missing_optional_evidence_handling(self, clean_explainability_environment):
        gen = get_explanation_generator()
        repo = get_wellness_repository()
        item = repo.get_by_id("well_001")
        state = analyze_emotional_state("Neutral feeling", model_type="distilbert")
        prof = clean_explainability_environment[1].get_or_create_profile("empty_user")

        # Pass 0.0 scores for optional history / feedback
        exp = gen.generate_explanation(
            item=item,
            emotional_state=state,
            user_profile=prof,
            tracked_user_state=None,
            e_rel=0.0,
            i_fit=0.0,
            p_match=0.0,
            sem_sim=0.0,
            h_score=0.0,
            hist_accept=0.0,
            hist_reject=0.0,
            rating_score=0.0,
            active_weights={},
        )
        assert isinstance(exp, RecommendationExplanation)
        assert len(exp.explanation_factors) == 6
        assert isinstance(exp.reason, str)
        assert len(exp.reason) > 0

    def test_convenience_function_returns_explanations(self):
        res = get_personalized_recommendations(
            text="I am stressed out about work deadlines.",
            user_id="convenience_exp_user",
            top_k=3,
        )
        assert res["is_valid"] is True
        for rec in res["recommendations"]:
            assert "recommendation" in rec
            assert "score" in rec
            assert "reason" in rec
            assert "explanation_factors" in rec
            assert len(rec["explanation_factors"]) == 6
