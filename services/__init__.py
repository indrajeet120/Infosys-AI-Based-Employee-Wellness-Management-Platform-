"""
Services package for Text Sentiment Analysis and Deep Emotion Classification.
"""
from services.ingestion import ingest_text, ingest_txt_file, ingest_csv_file, IngestedItem
from services.preprocessing import preprocess_text, TextPreprocessor
from services.sentiment import analyze_sentiment, VaderSentimentAnalyzer
from services.emotion import analyze_emotion, get_emotion_classifier, EmotionClassifier, EmotionPrediction
from services.evaluation import evaluate_emotion_model, compare_models
from services.isear_validation import validate_on_isear_benchmark
from services.reporting import (
    generate_sentiment_report,
    generate_complete_report,
    calculate_summary_stats,
    process_pipeline_items,
)

from services.intensity import (
    EmotionalState,
    analyze_emotional_state,
    calculate_emotional_intensity,
    calculate_polarity,
    detect_mixed_emotions,
    determine_severity_level,
)
from services.wellness_data import WellnessContent, WellnessContentRepository, get_wellness_repository
from services.user_profile import UserProfile, UserProfileManager, check_collaborative_filtering_availability
from services.semantic_matcher import SemanticMatcher, get_semantic_matcher
from services.hybrid_recommender import (
    RecommendationItem,
    HybridRecommendationEngine,
    get_recommendation_engine,
    get_personalized_recommendations,
)
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
from services.feedback_learning import (
    FeedbackRecord,
    FeedbackManager,
    get_feedback_manager,
)
from services.explainability import (
    ExplanationFactor,
    RecommendationExplanation,
    ExplanationGenerator,
    get_explanation_generator,
)
def __getattr__(name: str):
    if name in [
        "RecommendationEvaluator",
        "run_recommendation_evaluation",
        "BaselineRuleRecommender",
        "build_controlled_evaluation_dataset",
        "calculate_precision_at_k",
        "calculate_recall_at_k",
        "calculate_f1_at_k",
        "calculate_ndcg_at_k",
        "calculate_ground_truth_acceptance_proxy",
        "calculate_diversity_score",
    ]:
        import services.recommendation_eval as rec_eval
        return getattr(rec_eval, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "ingest_text",
    "ingest_txt_file",
    "ingest_csv_file",
    "IngestedItem",
    "preprocess_text",
    "TextPreprocessor",
    "analyze_sentiment",
    "VaderSentimentAnalyzer",
    "analyze_emotion",
    "get_emotion_classifier",
    "EmotionClassifier",
    "EmotionPrediction",
    "evaluate_emotion_model",
    "compare_models",
    "validate_on_isear_benchmark",
    "generate_sentiment_report",
    "generate_complete_report",
    "calculate_summary_stats",
    "process_pipeline_items",
    # Milestone 3
    "EmotionalState",
    "analyze_emotional_state",
    "calculate_emotional_intensity",
    "calculate_polarity",
    "detect_mixed_emotions",
    "determine_severity_level",
    "WellnessContent",
    "WellnessContentRepository",
    "get_wellness_repository",
    "UserProfile",
    "UserProfileManager",
    "check_collaborative_filtering_availability",
    "SemanticMatcher",
    "get_semantic_matcher",
    "RecommendationItem",
    "HybridRecommendationEngine",
    "get_recommendation_engine",
    "get_personalized_recommendations",
    # Task 6 - Trend Analysis & State Tracking
    "TrackedUserState",
    "calculate_emotion_frequency",
    "calculate_intensity_trends",
    "determine_dominant_emotions",
    "calculate_polarity_trends",
    "detect_repeated_patterns",
    "build_tracked_user_state",
    "calculate_historical_emotion_synergy",
    # Task 7 - Feedback Learning
    "FeedbackRecord",
    "FeedbackManager",
    "get_feedback_manager",
    # Task 8 - Recommendation Explainability
    "ExplanationFactor",
    "RecommendationExplanation",
    "ExplanationGenerator",
    "get_explanation_generator",
    # Task 9 - ML Validation & Performance Testing
    "RecommendationEvaluator",
    "run_recommendation_evaluation",
    "BaselineRuleRecommender",
    "build_controlled_evaluation_dataset",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_f1_at_k",
    "calculate_ndcg_at_k",
    "calculate_ground_truth_acceptance_proxy",
    "calculate_diversity_score",
]

