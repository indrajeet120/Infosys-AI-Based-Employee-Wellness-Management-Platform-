"""
Unit tests for Advanced Search & Filtering Engine (Milestone 4 Task 4).
"""

import pytest
from datetime import datetime, timedelta
from services.filtering import FilterEngine, SearchFilterCriteria


@pytest.fixture
def sample_emotion_records():
    now = datetime.now()
    return [
        {
            "timestamp": (now - timedelta(days=5)).isoformat(),
            "dominant_emotion": "Joy",
            "intensity": 0.85,
            "text": "Had a fantastic team celebration after project launch.",
            "sentiment": "Positive",
        },
        {
            "timestamp": (now - timedelta(days=2)).isoformat(),
            "dominant_emotion": "Fear",
            "intensity": 0.65,
            "text": "Anxious about tomorrow's executive presentation.",
            "sentiment": "Negative",
        },
        {
            "timestamp": now.isoformat(),
            "dominant_emotion": "Sadness",
            "intensity": 0.40,
            "text": "Feeling low and exhausted after long week.",
            "sentiment": "Negative",
        },
    ]


@pytest.fixture
def sample_recommendations():
    return [
        {
            "content_id": "REC_BREATH_01",
            "title": "Box Breathing Exercise",
            "activity_type": "breathing exercise",
            "intensity_level": 0.70,
            "duration": "5 mins",
            "difficulty": "Easy",
            "tags": ["stress", "anxiety", "calm"],
            "score": 0.92,
        },
        {
            "content_id": "REC_WALK_01",
            "title": "Mindful Walk",
            "activity_type": "physical activity",
            "intensity_level": 0.40,
            "duration": "15 mins",
            "difficulty": "Easy",
            "tags": ["energy", "outdoor"],
            "score": 0.75,
        },
        {
            "content_id": "REC_JOURNAL_01",
            "title": "Reflective Journaling",
            "activity_type": "journaling",
            "intensity_level": 0.50,
            "duration": "10 mins",
            "difficulty": "Medium",
            "tags": ["reflection", "mindset"],
            "score": 0.81,
        },
    ]


def test_filter_by_query(sample_emotion_records):
    criteria = SearchFilterCriteria(query="presentation")
    filtered = FilterEngine.filter_records(sample_emotion_records, criteria, text_key="text")
    assert len(filtered) == 1
    assert filtered[0]["dominant_emotion"] == "Fear"


def test_filter_by_emotion_category(sample_emotion_records):
    criteria = SearchFilterCriteria(emotions=["Joy", "Sadness"])
    filtered = FilterEngine.filter_records(sample_emotion_records, criteria, emotion_key="dominant_emotion")
    assert len(filtered) == 2
    emotions = [r["dominant_emotion"] for r in filtered]
    assert "Joy" in emotions
    assert "Sadness" in emotions
    assert "Fear" not in emotions


def test_filter_by_intensity_range(sample_emotion_records):
    criteria = SearchFilterCriteria(min_intensity=0.60, max_intensity=0.90)
    filtered = FilterEngine.filter_records(sample_emotion_records, criteria, intensity_key="intensity")
    assert len(filtered) == 2
    intensities = [r["intensity"] for r in filtered]
    assert 0.85 in intensities
    assert 0.65 in intensities


def test_filter_recommendations_by_difficulty(sample_recommendations):
    criteria = SearchFilterCriteria(difficulty="Easy")
    filtered = FilterEngine.filter_recommendations(sample_recommendations, criteria)
    assert len(filtered) == 2
    diffs = {r["difficulty"] for r in filtered}
    assert diffs == {"Easy"}


def test_filter_recommendations_by_activity_type(sample_recommendations):
    criteria = SearchFilterCriteria(activity_types=["journaling"])
    filtered = FilterEngine.filter_recommendations(sample_recommendations, criteria)
    assert len(filtered) == 1
    assert filtered[0]["content_id"] == "REC_JOURNAL_01"


def test_filter_summary_stats(sample_emotion_records):
    criteria = SearchFilterCriteria(emotions=["Fear", "Sadness"])
    filtered = FilterEngine.filter_records(sample_emotion_records, criteria, emotion_key="dominant_emotion")
    stats = FilterEngine.get_filter_summary(sample_emotion_records, filtered, criteria)
    assert stats["total_input"] == 3
    assert stats["matched_count"] == 2
    assert stats["filtered_out_count"] == 1
    assert stats["match_percentage"] == pytest.approx(66.67, 0.1)
