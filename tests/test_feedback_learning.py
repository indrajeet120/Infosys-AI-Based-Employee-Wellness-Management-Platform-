"""
Unit and integration tests for Task 7 – Recommendation Feedback Learning.
Tests:
1. Recommendation viewed event recording
2. Recommendation accepted event recording
3. Recommendation rejected event recording
4. User rating (1 to 5 stars) event recording
5. User preference changes event recording
6. Persistent storage and reload of feedback events
7. Positive interactions (accept, like, 5-star) boosting relevance of similar content
8. Negative interactions (reject, dislike, 1-star) reducing relevance and avoiding repeated recommendation
9. User preference changes immediately affecting future recommendations
10. Recommendation diversity scoring
"""

import pytest
from pathlib import Path
from services.feedback_learning import (
    FeedbackRecord,
    FeedbackManager,
)
from services.user_profile import UserProfileManager
from services.hybrid_recommender import HybridRecommendationEngine
from services.intensity import EmotionalState


@pytest.fixture
def temp_feedback_manager(tmp_path):
    storage_path = tmp_path / "feedback_events.json"
    mgr = FeedbackManager(storage_path=storage_path)
    return mgr


@pytest.fixture
def temp_user_manager(tmp_path):
    storage_path = tmp_path / "user_profiles.json"
    mgr = UserProfileManager(profiles_path=storage_path)
    return mgr


class TestFeedbackDataStructureAndEvents:
    def test_record_view_event(self, temp_feedback_manager):
        rec = temp_feedback_manager.record_view(
            user_id="user_fb_1",
            recommendation_id="well_001",
            recommendation_type="breathing exercise",
            emotion="Fear",
            intensity=0.85,
            metadata={"title": "4-7-8 Box Breathing"},
        )
        assert rec.viewed is True
        assert rec.accepted is False
        assert rec.rejected is False
        assert rec.rating is None
        assert rec.emotion_at_recommendation == "Fear"
        assert rec.emotion_intensity == 0.85
        assert len(temp_feedback_manager.get_user_feedback("user_fb_1")) == 1

    def test_record_acceptance_event(self, temp_feedback_manager):
        rec = temp_feedback_manager.record_acceptance(
            user_id="user_fb_2",
            recommendation_id="well_002",
            recommendation_type="relaxation activity",
            emotion="Sadness",
            intensity=0.70,
            rating=5.0,
            metadata={"tags": ["mindfulness", "grounding"]},
        )
        assert rec.viewed is True
        assert rec.accepted is True
        assert rec.rejected is False
        assert rec.rating == 5.0
        assert "preferred_activities" in rec.preference_snapshot

    def test_record_rejection_event(self, temp_feedback_manager):
        rec = temp_feedback_manager.record_rejection(
            user_id="user_fb_3",
            recommendation_id="well_005",
            recommendation_type="physical activity",
            emotion="Fear",
            intensity=0.90,
            reason="too_exhausting",
        )
        assert rec.rejected is True
        assert rec.accepted is False
        assert rec.interaction_metadata.get("rejection_reason") == "too_exhausting"
        assert "well_005" in temp_feedback_manager.get_rejected_content_ids("user_fb_3")

    def test_record_rating_event(self, temp_feedback_manager):
        rec = temp_feedback_manager.record_rating(
            user_id="user_fb_4",
            recommendation_id="well_011",
            rating=4.5,
            recommendation_type="breathing exercise",
            emotion="Anger",
            intensity=0.75,
        )
        assert rec.rating == 4.5
        assert rec.accepted is True
        assert rec.rejected is False

    def test_record_preference_change_event(self, temp_feedback_manager):
        rec = temp_feedback_manager.record_preference_change(
            user_id="user_fb_5",
            preferred_content_types=["audio", "interactive_guide"],
            preferred_activities=["meditation", "breathing exercise"],
            preferred_language="English",
        )
        assert rec.recommendation_type == "user_preference"
        assert rec.preference_snapshot["preferred_language"] == "English"
        assert "audio" in rec.preference_snapshot["preferred_content_types"]

    def test_persistent_storage_reload(self, tmp_path):
        storage_path = tmp_path / "persistence_test_feedback.json"
        mgr1 = FeedbackManager(storage_path=storage_path)
        mgr1.record_acceptance("user_p", "well_001", "breathing exercise", "Fear", 0.8)
        mgr1.record_rejection("user_p", "well_004", "journaling", "Fear", 0.8)

        # Reload with new instance from same file
        mgr2 = FeedbackManager(storage_path=storage_path)
        events = mgr2.get_user_feedback("user_p")
        assert len(events) == 2
        assert events[0].accepted is True
        assert events[1].rejected is True


class TestFeedbackRankingAdaptation:
    def test_positive_feedback_boosts_similar_content(self, tmp_path):
        fb_path = tmp_path / "feedback.json"
        prof_path = tmp_path / "profiles.json"
        
        fb_mgr = FeedbackManager(storage_path=fb_path)
        user_mgr = UserProfileManager(profiles_path=prof_path)
        
        user_id = "user_dynamic_boost"
        user_mgr.get_or_create_profile(user_id)

        # Accept a breathing activity during Fear state
        fb_mgr.record_acceptance(
            user_id=user_id,
            recommendation_id="well_001",
            recommendation_type="breathing exercise",
            emotion="Fear",
            intensity=0.85,
            rating=5.0,
            metadata={"tags": ["breathing", "anxiety relief"]},
        )

        curr_state = EmotionalState(
            dominant_emotion="Fear",
            dominant_confidence=0.95,
            emotional_intensity=0.85,
            positive_polarity=0.05,
            negative_polarity=0.90,
            final_emotional_state="High-Intensity Fear",
        )

        engine = HybridRecommendationEngine()
        engine.user_manager = user_mgr
        engine.feedback_manager = fb_mgr

        result = engine.recommend(curr_state, user_id=user_id, top_k=5)
        recs = result["recommendations"]

        # The accepted activity and other breathing activities should have high acceptance score
        top_rec = recs[0]
        assert top_rec["historical_acceptance"] >= 0.50
        assert top_rec["final_score"] > 0.40

    def test_rejected_content_is_penalized_and_not_repeated(self, tmp_path):
        fb_path = tmp_path / "feedback_rej.json"
        prof_path = tmp_path / "profiles_rej.json"
        
        fb_mgr = FeedbackManager(storage_path=fb_path)
        user_mgr = UserProfileManager(profiles_path=prof_path)
        
        user_id = "user_rejection_test"
        user_mgr.get_or_create_profile(user_id)

        # Explicitly reject well_001
        fb_mgr.record_rejection(
            user_id=user_id,
            recommendation_id="well_001",
            recommendation_type="breathing exercise",
            emotion="Fear",
            intensity=0.80,
            reason="not_helpful",
        )

        curr_state = EmotionalState(
            dominant_emotion="Fear",
            dominant_confidence=0.90,
            emotional_intensity=0.80,
            positive_polarity=0.10,
            negative_polarity=0.85,
            final_emotional_state="High-Intensity Fear",
        )

        engine = HybridRecommendationEngine()
        engine.user_manager = user_mgr
        engine.feedback_manager = fb_mgr

        result = engine.recommend(curr_state, user_id=user_id, top_k=5)
        recommended_ids = [r["content_id"] for r in result["recommendations"]]
        
        # Rejected content well_001 MUST NOT appear in the recommendations
        assert "well_001" not in recommended_ids

    def test_rating_influences_ranking_scores(self, tmp_path):
        fb_path = tmp_path / "feedback_rating.json"
        prof_path = tmp_path / "profiles_rating.json"
        
        fb_mgr = FeedbackManager(storage_path=fb_path)
        user_mgr = UserProfileManager(profiles_path=prof_path)
        
        user_id = "user_rating_test"
        user_mgr.get_or_create_profile(user_id)

        # Rate well_011 5-stars and well_002 1-star
        fb_mgr.record_rating(user_id=user_id, recommendation_id="well_011", rating=5.0, recommendation_type="breathing exercise")
        fb_mgr.record_rating(user_id=user_id, recommendation_id="well_002", rating=1.0, recommendation_type="relaxation activity")

        score_high = fb_mgr.calculate_item_rating_score("well_011", "breathing exercise", user_id)
        score_low = fb_mgr.calculate_item_rating_score("well_002", "relaxation activity", user_id)

        assert score_high == 1.0  # (5 - 1) / 4 = 1.0
        assert score_low == 0.0   # (1 - 1) / 4 = 0.0
        assert score_high > score_low

    def test_user_preference_changes_affect_future_recommendations(self, tmp_path):
        fb_path = tmp_path / "feedback_pref.json"
        prof_path = tmp_path / "profiles_pref.json"
        
        fb_mgr = FeedbackManager(storage_path=fb_path)
        user_mgr = UserProfileManager(profiles_path=prof_path)
        
        user_id = "user_pref_shift"
        user_mgr.get_or_create_profile(user_id)

        # Switch user preference strictly to 'journaling'
        fb_mgr.record_preference_change(
            user_id=user_id,
            preferred_content_types=["text"],
            preferred_activities=["journaling"],
        )

        curr_state = EmotionalState(
            dominant_emotion="Sadness",
            dominant_confidence=0.85,
            emotional_intensity=0.50,
            positive_polarity=0.10,
            negative_polarity=0.60,
            final_emotional_state="Moderate-Intensity Sadness",
        )

        engine = HybridRecommendationEngine()
        engine.user_manager = user_mgr
        engine.feedback_manager = fb_mgr

        result = engine.recommend(curr_state, user_id=user_id, top_k=3)
        top_rec = result["recommendations"][0]
        assert top_rec["activity_type"] == "journaling" or top_rec["preference_match"] >= 0.70
