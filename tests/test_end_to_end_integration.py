"""
End-to-End System Integration Test for MoodMentor (Milestone 4 Task 6).
Verifies complete pipeline from text input to ML scoring, recommendation generation,
interactive feedback logging, trend analysis, filtering, and report generation.
"""

import pytest
import os
from pathlib import Path
from services.ingestion import ingest_text
from services.preprocessing import preprocess_text
from services.sentiment import analyze_sentiment
from services.emotion import get_emotion_classifier
from services.intensity import analyze_emotional_state
from services.hybrid_recommender import get_personalized_recommendations
from services.user_profile import get_user_profile_manager
from services.feedback_learning import get_feedback_manager
from services.filtering import FilterEngine, SearchFilterCriteria
from services.reporting import export_records_to_csv, export_analytics_pdf


def test_full_end_to_end_pipeline_flow(tmp_path):
    user_id = "test_e2e_employee_99"
    raw_input = "I am extremely overwhelmed by the upcoming launch deadline and feeling anxious."

    # Step 1: Text Ingestion & Preprocessing (Milestone 1)
    ingested_list = ingest_text(raw_input, source="unit_test")
    ingested = ingested_list[0]
    assert ingested.is_valid is True
    cleaned_text = preprocess_text(ingested.text)
    assert len(cleaned_text) > 0

    # Step 2: VADER Sentiment Baseline (Milestone 1)
    sentiment_res = analyze_sentiment(cleaned_text)
    assert sentiment_res["sentiment"] in ["Positive", "Negative", "Neutral"]
    assert "compound" in sentiment_res

    # Step 3: Transformer Emotion Classification (Milestone 2)
    classifier = get_emotion_classifier("distilbert")
    predictions = classifier.predict(cleaned_text)
    assert len(predictions.probabilities) == 6
    assert predictions.primary_confidence > 0.0

    # Step 4: Emotion Intensity & Hybrid Recommendation Engine (Milestone 3)
    rec_result = get_personalized_recommendations(
        text=raw_input,
        user_id=user_id,
        model_type="distilbert",
        top_k=3,
    )
    assert rec_result["is_valid"] is True
    recs = rec_result["recommendations"]
    assert len(recs) == 3

    # Step 5: User Profile & Emotion History Recording (Milestone 3 & 4)
    prof_mgr = get_user_profile_manager()
    prof_mgr.record_emotion_entry(
        user_id=user_id,
        dominant_emotion=rec_result["emotional_state"]["dominant_emotion"],
        intensity=rec_result["emotional_state"]["emotional_intensity"],
        positive_polarity=rec_result["emotional_state"]["positive_polarity"],
        negative_polarity=rec_result["emotional_state"]["negative_polarity"],
        final_emotional_state=rec_result["emotional_state"]["final_emotional_state"],
        raw_text=raw_input,
    )
    prof = prof_mgr.get_or_create_profile(user_id)
    assert len(prof.emotion_history) >= 1

    # Step 6: Interactive Feedback Recording (Milestone 4 Task 3)
    fb_mgr = get_feedback_manager()
    first_rec = recs[0]
    fb_mgr.record_acceptance(
        user_id=user_id,
        recommendation_id=first_rec["content_id"],
        recommendation_type=first_rec["activity_type"],
        emotion=rec_result["emotional_state"]["dominant_emotion"],
        intensity=rec_result["emotional_state"]["emotional_intensity"],
        rating=5.0,
    )
    accepted_ids = fb_mgr.get_accepted_content_ids(user_id)
    assert first_rec["content_id"] in accepted_ids

    # Step 7: Filtering Engine Validation (Milestone 4 Task 4)
    criteria = SearchFilterCriteria(query="overwhelmed")
    filtered_history = FilterEngine.filter_records(prof.emotion_history, criteria, text_key="raw_text")
    assert len(filtered_history) >= 1

    # Step 8: CSV & PDF Report Export (Milestone 4 Task 5)
    csv_file = tmp_path / "report.csv"
    pdf_file = tmp_path / "report.pdf"

    export_records_to_csv([rec_result["emotional_state"]], str(csv_file))
    assert csv_file.exists()
    assert os.path.getsize(csv_file) > 0

    analytics_summary = {
        "user_id": user_id,
        "total_checkins": len(prof.emotion_history),
        "dominant_emotion": rec_result["emotional_state"]["dominant_emotion"],
        "avg_intensity": rec_result["emotional_state"]["emotional_intensity"],
        "total_feedback": 1,
        "acceptance_rate": 100.0,
    }

    pdf_path = export_analytics_pdf(
        user_id=user_id,
        analytics_summary=analytics_summary,
        emotion_records=prof.emotion_history,
        recommendations=recs,
        output_filepath=str(pdf_file),
    )
    assert Path(pdf_path).exists()
    assert os.path.getsize(pdf_path) > 1000
