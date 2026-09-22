"""
End-to-end integration tests for Milestone 3:
Text -> Emotion Model -> Emotional State -> User Profile -> Candidate Generation -> Semantic Matching -> Ranking -> Final Recommendation -> Feedback Loop.
"""

from services.intensity import analyze_emotional_state
from services.hybrid_recommender import get_recommendation_engine, get_personalized_recommendations
from services.user_profile import get_user_profile_manager


class TestMilestone3Integration:
    def test_full_pipeline_joy_scenario(self):
        text = "I received my dream job offer today and cannot stop smiling!"
        user_id = "integration_user_joy"
        
        # 1. Pipeline Execution
        result = get_personalized_recommendations(text, user_id=user_id, top_k=3)
        assert result["is_valid"] is True
        
        # 2. Emotional State Checks
        state = result["emotional_state"]
        assert state["dominant_emotion"] == "Joy"
        assert state["positive_polarity"] > state["negative_polarity"]
        assert 0.0 <= state["emotional_intensity"] <= 1.0
        
        # 3. Recommendations Check
        recs = result["recommendations"]
        assert len(recs) == 3
        top_rec = recs[0]
        assert "joy" in top_rec["target_emotions"] or top_rec["final_score"] > 0.4
        
        # 4. Feedback Loop Check
        user_mgr = get_user_profile_manager()
        user_mgr.record_feedback(
            user_id=user_id,
            content_id=top_rec["content_id"],
            feedback_type="like",
            recommendation_score=top_rec["final_score"],
        )
        profile = user_mgr.get_or_create_profile(user_id)
        assert top_rec["content_id"] in profile.liked_content
        assert len(profile.emotion_history) >= 1

    def test_full_pipeline_mixed_anxiety_scenario(self):
        text = "I am excited about the new job but really terrified of failing."
        user_id = "integration_user_mixed"
        
        result = get_personalized_recommendations(text, user_id=user_id, model_type="bert", top_k=4)
        assert result["is_valid"] is True
        state = result["emotional_state"]
        
        assert state["dominant_emotion"] in ["Fear", "Joy"]
        assert 0.0 <= state["emotional_intensity"] <= 1.0
        assert isinstance(state["mixed_emotion"], bool)
        
        recs = result["recommendations"]
        assert len(recs) == 4
        # Verify descending ranking
        for i in range(len(recs) - 1):
            assert recs[i]["final_score"] >= recs[i + 1]["final_score"]

    def test_invalid_input_fails_gracefully_without_crashing(self):
        result = get_personalized_recommendations("   \n\t   ", user_id="invalid_user")
        assert result["is_valid"] is False
        assert len(result["recommendations"]) == 0
        assert "error_message" in result
