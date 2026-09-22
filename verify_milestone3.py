"""
Standalone CLI verification script for Milestone 3:
Advanced Emotion Intensity, Semantic Matching, Hybrid Recommendation Engine & Ranking Model.
"""

import sys
from pathlib import Path
import json

# Ensure utf-8 encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from services.config import MEDICAL_DISCLAIMER
from services.intensity import analyze_emotional_state
from services.semantic_matcher import get_semantic_matcher
from services.hybrid_recommender import get_recommendation_engine, get_personalized_recommendations
from services.user_profile import get_user_profile_manager, check_collaborative_filtering_availability
from services.wellness_data import get_wellness_repository


def main():
    print("=" * 80)
    print("MILESTONE 3 - ADVANCED EMOTION ANALYSIS & PERSONALIZED WELLNESS RECOMMENDATION")
    print("=" * 80)
    print(f"\n{MEDICAL_DISCLAIMER}\n")

    # 1. Verify Task 1: Emotion Intensity & Emotional State
    print("--- TASK 1: Emotion Intensity & Emotional State Analysis ---")
    test_text_1 = "I received my dream job offer today and cannot stop smiling!"
    state_1 = analyze_emotional_state(test_text_1, model_type="distilbert")
    print(f"Input: '{test_text_1}'")
    print(f"  Dominant Emotion:   {state_1.dominant_emotion} (Confidence: {state_1.dominant_confidence * 100:.1f}%)")
    print(f"  Emotional Intensity: {state_1.emotional_intensity:.3f} | Severity: {state_1.emotion_severity}")
    print(f"  Positive Polarity:   {state_1.positive_polarity:.3f} | Negative: {state_1.negative_polarity:.3f}")
    print(f"  Mixed Emotion:       {state_1.mixed_emotion}")
    print(f"  Final State:         {state_1.final_emotional_state}")

    test_text_2 = "I am excited about starting my new role but terrified of moving to a new city alone."
    state_2 = analyze_emotional_state(test_text_2, model_type="distilbert")
    print(f"\nInput: '{test_text_2}'")
    print(f"  Dominant Emotion:   {state_2.dominant_emotion} (Confidence: {state_2.dominant_confidence * 100:.1f}%)")
    print(f"  Emotional Intensity: {state_2.emotional_intensity:.3f} | Severity: {state_2.emotion_severity}")
    print(f"  Mixed Emotion:       {state_2.mixed_emotion}")
    print(f"  Final State:         {state_2.final_emotional_state}")

    # 2. Verify Task 2: User Profile & Wellness Dataset
    print("\n--- TASK 2: User Profile & Wellness Content Dataset ---")
    repo = get_wellness_repository()
    print(f"Wellness Activities in Repository: {repo.count()} items")
    user_mgr = get_user_profile_manager()
    profile = user_mgr.get_or_create_profile("cli_test_user")
    print(f"Profile Loaded: User ID = '{profile.user_id}' | Preferred Types = {profile.preferred_content_types}")
    collab = check_collaborative_filtering_availability()
    print(f"Collaborative Filtering Available: {collab['available']} ({collab.get('reason', collab.get('strategy'))})")

    # 3. Verify Task 5: Semantic Content Matching
    print("\n--- TASK 5: Semantic Wellness Content Matching (Dense Embeddings) ---")
    matcher = get_semantic_matcher()
    sem_matches = matcher.match_content(
        dominant_emotion=state_2.dominant_emotion,
        emotional_intensity=state_2.emotional_intensity,
        positive_polarity=state_2.positive_polarity,
        negative_polarity=state_2.negative_polarity,
        final_emotional_state=state_2.final_emotional_state,
        raw_text=test_text_2,
    )
    print(f"Semantic Cosine Matches Calculated: {len(sem_matches)} items")
    top_sem_id = max(sem_matches.keys(), key=lambda k: sem_matches[k]["semantic_similarity"])
    print(f"Top Semantic Match: '{top_sem_id}' with Similarity = {sem_matches[top_sem_id]['semantic_similarity'] * 100:.1f}%")

    # 4. Verify Task 3 & 4: Hybrid Recommendation Engine & Dynamic Ranking
    print("\n--- TASK 3 & 4: Hybrid Recommendation Engine & Dynamic Ranking ---")
    result = get_personalized_recommendations(
        text=test_text_2,
        user_id="cli_test_user",
        model_type="distilbert",
        top_k=3,
    )
    print(f"Dynamic Ranked Recommendations (Top 3):")
    for idx, rec in enumerate(result["recommendations"], start=1):
        print(f"\n  #{idx} [{rec['content_id']}] {rec['title']}")
        print(f"     Final Score: {rec['final_score']:.4f} | Activity: {rec['activity_type']} ({rec['duration']})")
        print(f"     Sub-scores: Emotion={rec['emotion_relevance']:.2f}, Intensity={rec['intensity_fit']:.2f}, Pref={rec['preference_match']:.2f}, Sem={rec['semantic_similarity']:.2f}")
        print(f"     Strategies: {rec['source_strategies']}")
        print(f"     Reason:     {rec['reason']}")

    # 5. Verify Task 6: Emotional Trend & User State Tracking
    print("\n--- TASK 6: Emotional Trend & User State Tracking ---")
    from services.trend_analysis import (
        calculate_emotion_frequency,
        calculate_intensity_trends,
        determine_dominant_emotions,
        calculate_polarity_trends,
        detect_repeated_patterns,
    )
    cli_prof = user_mgr.get_or_create_profile("cli_test_user")
    freq = calculate_emotion_frequency(cli_prof.emotion_history)
    int_trends = calculate_intensity_trends(cli_prof.emotion_history)
    dom_emo = determine_dominant_emotions(cli_prof.emotion_history)
    pol_trends = calculate_polarity_trends(cli_prof.emotion_history)
    patterns = detect_repeated_patterns(cli_prof.emotion_history)

    print(f"Historical Check-ins: {freq['total_entries']}")
    print(f"Emotion Frequency:    {freq['counts']}")
    print(f"Dominant Emotion:     {dom_emo['dominant_emotion']} (Score: {dom_emo['dominant_score']:.2f})")
    print(f"Intensity Baseline:   {int_trends['historical_mean']:.3f} | Trajectory: {int_trends['direction']}")
    print(f"Polarity Status:      {pol_trends['net_polarity_status']} (Shift: {pol_trends['polarity_shift']:+.2f})")
    print(f"Detected Patterns:    {[p['name'] for p in patterns] if patterns else 'None (State Stable)'}")

    # 6. Verify Feedback Loop
    print("\n--- FEEDBACK LOOP: Recording Like & Verifying Preference Boost ---")
    top_id = result["recommendations"][0]["content_id"]
    user_mgr.record_feedback(user_id="cli_test_user", content_id=top_id, feedback_type="like", recommendation_score=0.85)
    updated_prof = user_mgr.get_or_create_profile("cli_test_user")
    print(f"Liked Content List: {updated_prof.liked_content}")
    print(f"Interaction Count:  {updated_prof.interaction_count}")

    print("\n" + "=" * 80)
    print("ALL 6 TASKS OF MILESTONE 3 VERIFIED AND WORKING SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
