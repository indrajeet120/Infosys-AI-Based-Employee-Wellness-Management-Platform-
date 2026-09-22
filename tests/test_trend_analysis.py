"""
Unit and integration tests for Task 6 – Emotional Trend & User State Tracking.
Tests:
1. Emotion Frequency (all 6 emotions, counts, proportions)
2. Emotion Intensity Over Time (moving average, historical mean, delta, direction)
3. Dominant Emotions (recency-decay weighted scoring, not just latest)
4. Positive and Negative Polarity Trends (baseline vs recent, slopes, shifts)
5. Repeated Emotional Patterns (streaks, chronic high intensity, transitions, volatility)
6. Recent Emotional State (structured TrackedUserState object)
7. Recommendation Integration (historical feedback and emotional patterns dynamically affecting ranking)
"""

import pytest
from datetime import datetime, timezone
from services.trend_analysis import (
    TrackedUserState,
    calculate_emotion_frequency,
    calculate_intensity_trends,
    determine_dominant_emotions,
    calculate_polarity_trends,
    detect_repeated_patterns,
    build_tracked_user_state,
    calculate_historical_emotion_synergy,
)
from services.intensity import EmotionalState
from services.user_profile import UserProfileManager, UserProfile
from services.hybrid_recommender import HybridRecommendationEngine, get_personalized_recommendations


@pytest.fixture
def sample_emotion_history():
    """Generates realistic synthetic multi-session emotional history records."""
    return [
        {
            "timestamp": "2026-09-01T10:00:00Z",
            "dominant_emotion": "Joy",
            "intensity": 0.55,
            "positive_polarity": 0.85,
            "negative_polarity": 0.05,
            "final_emotional_state": "Moderate-Intensity Joy",
        },
        {
            "timestamp": "2026-09-02T10:00:00Z",
            "dominant_emotion": "Fear",
            "intensity": 0.78,
            "positive_polarity": 0.10,
            "negative_polarity": 0.82,
            "final_emotional_state": "High-Intensity Fear",
        },
        {
            "timestamp": "2026-09-03T10:00:00Z",
            "dominant_emotion": "Fear",
            "intensity": 0.82,
            "positive_polarity": 0.08,
            "negative_polarity": 0.88,
            "final_emotional_state": "High-Intensity Fear",
        },
        {
            "timestamp": "2026-09-04T10:00:00Z",
            "dominant_emotion": "Fear",
            "intensity": 0.85,
            "positive_polarity": 0.05,
            "negative_polarity": 0.90,
            "final_emotional_state": "Very High-Intensity Fear",
        },
        {
            "timestamp": "2026-09-05T10:00:00Z",
            "dominant_emotion": "Sadness",
            "intensity": 0.65,
            "positive_polarity": 0.12,
            "negative_polarity": 0.70,
            "final_emotional_state": "High-Intensity Sadness",
        },
    ]


class TestEmotionFrequency:
    def test_frequency_calculation_all_six_emotions(self):
        history = [
            {"dominant_emotion": "Joy", "intensity": 0.5},
            {"dominant_emotion": "Joy", "intensity": 0.6},
            {"dominant_emotion": "Sadness", "intensity": 0.4},
            {"dominant_emotion": "Anger", "intensity": 0.7},
            {"dominant_emotion": "Fear", "intensity": 0.8},
            {"dominant_emotion": "Surprise", "intensity": 0.3},
            {"dominant_emotion": "Disgust", "intensity": 0.5},
        ]
        res = calculate_emotion_frequency(history)
        counts = res["counts"]
        proportions = res["proportions"]
        
        assert res["total_entries"] == 7
        assert counts["Joy"] == 2
        assert counts["Sadness"] == 1
        assert counts["Anger"] == 1
        assert counts["Fear"] == 1
        assert counts["Surprise"] == 1
        assert counts["Disgust"] == 1
        assert pytest.approx(sum(proportions.values()), 0.01) == 1.0
        assert res["ranked_emotions"][0]["emotion"] == "Joy"

    def test_frequency_empty_history(self):
        res = calculate_emotion_frequency([])
        assert res["total_entries"] == 0
        assert res["counts"]["Joy"] == 0
        assert res["proportions"]["Fear"] == 0.0

    def test_frequency_window_size(self):
        history = [
            {"dominant_emotion": "Joy", "intensity": 0.5},
            {"dominant_emotion": "Joy", "intensity": 0.5},
            {"dominant_emotion": "Fear", "intensity": 0.8},
        ]
        res = calculate_emotion_frequency(history, window_size=1)
        assert res["total_entries"] == 1
        assert res["counts"]["Fear"] == 1
        assert res["counts"]["Joy"] == 0


class TestIntensityTrends:
    def test_intensity_escalating_trend(self):
        history = [
            {"intensity": 0.30, "timestamp": "T1"},
            {"intensity": 0.35, "timestamp": "T2"},
            {"intensity": 0.75, "timestamp": "T3"},
            {"intensity": 0.85, "timestamp": "T4"},
        ]
        res = calculate_intensity_trends(history, window_size=2)
        assert res["current_intensity"] == 0.85
        assert res["recent_mean"] == 0.80
        assert res["historical_mean"] == pytest.approx(0.5625, 0.01)
        assert res["delta"] > 0
        assert res["direction"] == "escalating"

    def test_intensity_decreasing_trend(self):
        history = [
            {"intensity": 0.90, "timestamp": "T1"},
            {"intensity": 0.85, "timestamp": "T2"},
            {"intensity": 0.30, "timestamp": "T3"},
            {"intensity": 0.25, "timestamp": "T4"},
        ]
        res = calculate_intensity_trends(history, window_size=2)
        assert res["direction"] == "decreasing"
        assert res["delta"] < -0.08

    def test_intensity_empty(self):
        res = calculate_intensity_trends([])
        assert res["current_intensity"] == 0.0
        assert res["direction"] == "stable"


class TestDominantEmotion:
    def test_dominant_emotion_does_not_just_pick_latest(self):
        # 4 consecutive Fear entries followed by 1 Joy entry
        history = [
            {"dominant_emotion": "Fear", "intensity": 0.90},
            {"dominant_emotion": "Fear", "intensity": 0.85},
            {"dominant_emotion": "Fear", "intensity": 0.88},
            {"dominant_emotion": "Fear", "intensity": 0.82},
            {"dominant_emotion": "Joy", "intensity": 0.40},  # Latest, but minority
        ]
        res = determine_dominant_emotions(history, decay_factor=0.85)
        # Dominant should be Fear based on weighted historical evidence, NOT simply Joy
        assert res["dominant_emotion"] == "Fear"
        assert res["dominant_score"] > 1.0


class TestPolarityTrends:
    def test_polarity_trends_worsening_and_recovering(self, sample_emotion_history):
        res = calculate_polarity_trends(sample_emotion_history, recent_window=2)
        assert "positive_trend" in res
        assert "negative_trend" in res
        assert "polarity_shift" in res
        assert res["negative_trend"]["recent_mean"] > res["positive_trend"]["recent_mean"]


class TestRepeatedPatterns:
    def test_detect_persistent_streak(self):
        history = [
            {"dominant_emotion": "Sadness", "intensity": 0.60},
            {"dominant_emotion": "Fear", "intensity": 0.75},
            {"dominant_emotion": "Fear", "intensity": 0.80},
            {"dominant_emotion": "Fear", "intensity": 0.85},
        ]
        patterns = detect_repeated_patterns(history, min_streak=2)
        assert len(patterns) >= 1
        streak_pattern = next((p for p in patterns if p["pattern_type"] == "emotional_streak"), None)
        assert streak_pattern is not None
        assert streak_pattern["emotion"] == "Fear"
        assert streak_pattern["streak_length"] == 3

    def test_detect_chronic_high_intensity(self):
        history = [
            {"dominant_emotion": "Fear", "intensity": 0.80},
            {"dominant_emotion": "Anger", "intensity": 0.75},
            {"dominant_emotion": "Sadness", "intensity": 0.85},
        ]
        patterns = detect_repeated_patterns(history)
        chronic = next((p for p in patterns if p["pattern_type"] == "chronic_high_intensity"), None)
        assert chronic is not None
        assert chronic["severity"] == "High"


class TestTrackedUserState:
    def test_build_tracked_user_state_structure(self, sample_emotion_history):
        curr_state = EmotionalState(
            dominant_emotion="Fear",
            dominant_confidence=0.92,
            emotional_intensity=0.85,
            positive_polarity=0.05,
            negative_polarity=0.90,
            final_emotional_state="Very High-Intensity Fear",
        )
        tracked = build_tracked_user_state(curr_state, sample_emotion_history)
        
        # Verify all 8 required fields
        assert tracked.dominant_emotion == "Fear"
        assert tracked.current_emotion == "Fear"
        assert 0.0 <= tracked.current_intensity <= 1.0
        assert isinstance(tracked.recent_emotion_distribution, dict)
        assert "direction" in tracked.positive_trend
        assert "direction" in tracked.negative_trend
        assert isinstance(tracked.repeated_patterns, list)
        assert 0.0 <= tracked.confidence <= 1.0


class TestHistoricalRecommendationIntegration:
    def test_historical_synergy_boosts_relevant_activity(self, tmp_path):
        profiles_file = tmp_path / "user_profiles.json"
        mgr = UserProfileManager(profiles_path=profiles_file)
        
        user_id = "test_synergy_user"
        profile = mgr.get_or_create_profile(user_id)
        
        # Simulate user previously liking breathing exercises during high fear check-ins
        mgr.record_feedback(
            user_id=user_id,
            content_id="well_001",
            feedback_type="like",
            recommendation_score=0.80,
            dominant_emotion="Fear",
            intensity=0.85,
            activity_type="breathing exercise",
            tags=["breathing", "anxiety relief", "grounding"],
        )
        
        curr_state = EmotionalState(
            dominant_emotion="Fear",
            dominant_confidence=0.95,
            emotional_intensity=0.80,
            positive_polarity=0.05,
            negative_polarity=0.90,
            final_emotional_state="High-Intensity Fear",
            raw_text="I am overwhelmed with panic and fear.",
        )
        
        engine = HybridRecommendationEngine()
        engine.user_manager = mgr
        
        rec_res = engine.recommend(curr_state, user_id=user_id, top_k=3)
        assert "tracked_user_state" in rec_res
        assert len(rec_res["recommendations"]) == 3
        
        top_rec = rec_res["recommendations"][0]
        # Top recommendation should have high score and reflect calming/breathing fit
        assert top_rec["final_score"] > 0.40
        assert "historical_preference" in top_rec["source_strategies"] or top_rec["activity_type"] in ["breathing exercise", "relaxation activity"]
