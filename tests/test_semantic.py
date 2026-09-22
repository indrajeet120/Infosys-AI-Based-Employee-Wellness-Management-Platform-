"""
Unit tests for Task 5: Semantic Wellness Content Matching.
"""

from pathlib import Path
import numpy as np
import pytest
from services.semantic_matcher import SemanticMatcher, get_semantic_matcher
from services.config import WELLNESS_EMBEDDINGS_PATH


class TestSemanticMatcher:
    def test_semantic_matcher_initialization_and_cache(self):
        matcher = get_semantic_matcher()
        assert matcher is not None
        assert matcher._is_initialized is True
        assert matcher._content_embeddings is not None
        assert len(matcher._content_embeddings) > 0
        assert WELLNESS_EMBEDDINGS_PATH.exists()

    def test_embedding_generation_shape_and_normalization(self):
        matcher = get_semantic_matcher()
        sample_texts = ["Mindful breathing exercise to reduce anxiety.", "Joyful celebration of daily success."]
        embeddings = matcher.encode(sample_texts)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 384  # Standard MiniLM embedding dimension
        
        # Verify L2 normalization: norm should be ~1.0
        norms = np.linalg.norm(embeddings, axis=1)
        for n in norms:
            assert np.isclose(n, 1.0, atol=1e-3)

    def test_cosine_similarity_score_range(self):
        matcher = get_semantic_matcher()
        results = matcher.match_content(
            dominant_emotion="Fear",
            emotional_intensity=0.82,
            positive_polarity=0.05,
            negative_polarity=0.85,
            final_emotional_state="High-Intensity Fear",
            raw_text="I am terrified about the upcoming medical procedure.",
        )
        assert len(results) > 0
        for cid, data in results.items():
            score = data["semantic_similarity"]
            assert 0.0 <= score <= 1.0
            assert "matching_reason" in data

    def test_semantic_differentiation(self):
        matcher = get_semantic_matcher()
        
        # Query for high panic/fear
        fear_results = matcher.match_content(
            dominant_emotion="Fear",
            emotional_intensity=0.90,
            positive_polarity=0.0,
            negative_polarity=0.90,
            final_emotional_state="Very High Intensity Fear",
            raw_text="Panic attack and racing heart rate.",
        )
        
        # Query for cheerful joy
        joy_results = matcher.match_content(
            dominant_emotion="Joy",
            emotional_intensity=0.80,
            positive_polarity=0.90,
            negative_polarity=0.0,
            final_emotional_state="High Intensity Joy",
            raw_text="I am so cheerful, grateful, and celebrating my promotion!",
        )

        # 4-7-8 breathing (well_001) should match fear higher than joy
        assert fear_results["well_001"]["semantic_similarity"] > 0.45
        # Joy savoring (well_005) should match joy higher than fear
        assert joy_results["well_005"]["semantic_similarity"] > fear_results["well_005"]["semantic_similarity"]
