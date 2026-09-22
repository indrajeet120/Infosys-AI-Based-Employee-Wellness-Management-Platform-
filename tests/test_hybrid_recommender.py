"""
Unit tests for Task 3 & Task 4: Hybrid Recommendation Engine & Dynamic Ranking Model.
"""

import pytest
from services.intensity import analyze_emotional_state
from services.hybrid_recommender import (
    HybridRecommendationEngine,
    get_recommendation_engine,
    get_personalized_recommendations,
)
from services.user_profile import UserProfileManager


@pytest.fixture
def clean_user_manager(tmp_path):
    p_path = tmp_path / "test_profiles.json"
    i_path = tmp_path / "test_interactions.json"
    return UserProfileManager(profiles_path=p_path, interactions_path=i_path)


class TestHybridRecommender:
    def test_recommendation_generation_and_order(self):
        engine = get_recommendation_engine()
        state = analyze_emotional_state("I am having severe anxiety and panic before my speech.", model_type="distilbert")
        
        result = engine.recommend(emotional_state=state, user_id="test_user_1", top_k=5)
        recs = result["recommendations"]
        assert len(recs) == 5

        # Check strictly descending order
        scores = [r["final_score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommendation_fields_and_sub_scores(self):
        engine = get_recommendation_engine()
        state = analyze_emotional_state("I am furious and my blood is boiling!", model_type="distilbert")
        
        result = engine.recommend(emotional_state=state, user_id="test_user_2", top_k=3)
        for r in result["recommendations"]:
            assert "content_id" in r
            assert "title" in r
            assert "final_score" in r
            assert "emotion_relevance" in r
            assert "intensity_fit" in r
            assert "preference_match" in r
            assert "semantic_similarity" in r
            assert "source_strategies" in r
            assert len(r["source_strategies"]) > 0
            assert "reason" in r
            assert len(r["reason"]) > 0
            assert 0.0 <= r["final_score"] <= 1.0

    def test_disliked_content_penalty_and_filtering(self):
        engine = get_recommendation_engine()
        user_mgr = engine.user_manager
        
        # User dislikes well_001
        user_mgr.record_feedback(user_id="test_dislike_user", content_id="well_001", feedback_type="dislike")
        
        state = analyze_emotional_state("I feel anxious and scared.", model_type="distilbert")
        result = engine.recommend(emotional_state=state, user_id="test_dislike_user", top_k=10)
        
        rec_ids = [r["content_id"] for r in result["recommendations"]]
        assert "well_001" not in rec_ids

    def test_dynamic_re_ranking_after_preference_change(self):
        engine = get_recommendation_engine()
        user_mgr = engine.user_manager
        
        # User updates preferences to prefer 'journaling'
        user_mgr.update_preferences(
            user_id="pref_user",
            preferred_content_types=["text"],
            preferred_activities=["journaling"],
        )
        
        state = analyze_emotional_state("I feel deep sorrow and heartbreak.", model_type="distilbert")
        result = engine.recommend(emotional_state=state, user_id="pref_user", top_k=5)
        
        activities = [r["activity_type"] for r in result["recommendations"]]
        assert "journaling" in activities

    def test_convenience_function_get_personalized_recommendations(self):
        res = get_personalized_recommendations(
            text="I am thrilled about winning the contest!",
            user_id="happy_user",
            top_k=4,
        )
        assert res["is_valid"] is True
        assert len(res["recommendations"]) == 4
        assert res["emotional_state"]["dominant_emotion"] == "Joy"
