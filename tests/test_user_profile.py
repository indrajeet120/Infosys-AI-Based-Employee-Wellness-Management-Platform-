"""
Unit tests for Task 2: User Profile, Preferences, Interaction History & Feedback.
"""

from pathlib import Path
import pytest
from services.user_profile import (
    UserProfile,
    UserProfileManager,
    check_collaborative_filtering_availability,
)
from services.wellness_data import get_wellness_repository


@pytest.fixture
def temp_user_manager(tmp_path):
    p_path = tmp_path / "test_user_profiles.json"
    i_path = tmp_path / "test_interactions.json"
    return UserProfileManager(profiles_path=p_path, interactions_path=i_path)


class TestUserProfile:
    def test_create_and_retrieve_profile(self, temp_user_manager):
        profile = temp_user_manager.get_or_create_profile("user_101")
        assert profile.user_id == "user_101"
        assert isinstance(profile.preferred_content_types, list)
        assert isinstance(profile.preferred_activities, list)

    def test_update_user_preferences(self, temp_user_manager):
        temp_user_manager.update_preferences(
            user_id="user_102",
            preferred_content_types=["audio", "exercise"],
            preferred_activities=["meditation", "breathing exercise"],
            preferred_language="English",
        )
        updated = temp_user_manager.get_or_create_profile("user_102")
        assert updated.preferred_content_types == ["audio", "exercise"]
        assert updated.preferred_activities == ["meditation", "breathing exercise"]
        assert updated.preferred_language == "English"

    def test_record_emotion_history(self, temp_user_manager):
        temp_user_manager.record_emotion_entry(
            user_id="user_103",
            dominant_emotion="Sadness",
            intensity=0.72,
            positive_polarity=0.10,
            negative_polarity=0.78,
            final_emotional_state="High-Intensity Sadness",
            raw_text="I feel really lonely today.",
        )
        profile = temp_user_manager.get_or_create_profile("user_103")
        assert len(profile.emotion_history) == 1
        assert len(profile.recent_emotions) == 1
        assert profile.recent_emotions[0] == "Sadness"
        assert profile.emotion_history[0]["intensity"] == 0.72

    def test_record_feedback_like_dislike_loop(self, temp_user_manager):
        temp_user_manager.record_feedback(
            user_id="user_104",
            content_id="well_001",
            feedback_type="like",
            recommendation_score=0.88,
        )
        profile = temp_user_manager.get_or_create_profile("user_104")
        assert "well_001" in profile.liked_content
        assert "well_001" not in profile.disliked_content
        assert profile.interaction_count == 1

        # Now dislike another content
        temp_user_manager.record_feedback(
            user_id="user_104",
            content_id="well_002",
            feedback_type="dislike",
            recommendation_score=0.45,
        )
        profile_after = temp_user_manager.get_or_create_profile("user_104")
        assert "well_002" in profile_after.disliked_content
        assert profile_after.interaction_count == 2

    def test_collaborative_filtering_status_and_fallback(self):
        status = check_collaborative_filtering_availability(min_users=100, min_interactions=500)
        assert status["available"] is False
        assert "fallback_strategy" in status
        assert "reason" in status

    def test_wellness_content_repository_loading(self):
        repo = get_wellness_repository()
        assert repo.count() >= 20
        all_items = repo.get_all()
        first = all_items[0]
        assert hasattr(first, "content_id")
        assert hasattr(first, "title")
        assert hasattr(first, "activity_type")
        assert hasattr(first, "target_emotions")
