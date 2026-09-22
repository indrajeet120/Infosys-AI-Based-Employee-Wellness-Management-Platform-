"""
Semantic Wellness Content Matching Service.
Computes dense sentence embeddings for user emotional context and wellness content,
calculating cosine similarity using a cached Transformer embedding model.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import numpy as np
import torch

from services.config import (
    WELLNESS_EMBEDDINGS_PATH,
    WELLNESS_EMBEDDINGS_META_PATH,
    EMBEDDING_MODEL_NAME,
    get_device,
)
from services.wellness_data import WellnessContent, get_wellness_repository


class SemanticMatcher:
    """Manages embedding generation, caching, and semantic cosine matching."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._content_embeddings: Optional[np.ndarray] = None
        self._content_ids: List[str] = []
        self._is_initialized = False

    def load_model(self) -> None:
        """Loads and caches the SentenceTransformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=str(get_device()))
            except Exception:
                # Fallback to local transformers AutoModel with mean-pooling
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.transformer_model = AutoModel.from_pretrained(self.model_name).to(get_device())
                self.model = "transformers_fallback"

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generates normalized L2 embeddings for a list of texts."""
        self.load_model()
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        if hasattr(self.model, "encode"):
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            return embeddings.astype(np.float32)
        else:
            # Fallback transformer mean-pooling
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(get_device())
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                embeddings = torch.sum(outputs.last_hidden_state * attention_mask, dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                return embeddings.cpu().numpy().astype(np.float32)

    def initialize_content_embeddings(self, force_recompute: bool = False) -> None:
        """Precomputes and caches embeddings for all wellness content items."""
        repo = get_wellness_repository()
        items = repo.get_all()
        current_ids = [item.content_id for item in items]

        # Check if persisted embeddings are valid
        if not force_recompute and WELLNESS_EMBEDDINGS_PATH.exists() and WELLNESS_EMBEDDINGS_META_PATH.exists():
            try:
                with open(WELLNESS_EMBEDDINGS_META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("content_ids") == current_ids:
                    self._content_embeddings = np.load(WELLNESS_EMBEDDINGS_PATH)
                    self._content_ids = current_ids
                    self._is_initialized = True
                    return
            except Exception:
                pass

        # Compute embeddings in batch
        texts_to_embed = [item.to_embedding_text() for item in items]
        embeddings = self.encode(texts_to_embed)

        # Save to disk
        WELLNESS_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(WELLNESS_EMBEDDINGS_PATH, embeddings)
        with open(WELLNESS_EMBEDDINGS_META_PATH, "w", encoding="utf-8") as f:
            json.dump({"content_ids": current_ids, "model_name": self.model_name}, f)

        self._content_embeddings = embeddings
        self._content_ids = current_ids
        self._is_initialized = True

    def build_emotional_query_text(
        self,
        dominant_emotion: str,
        emotional_intensity: float,
        positive_polarity: float,
        negative_polarity: float,
        final_emotional_state: str,
        raw_text: str = "",
    ) -> str:
        """Constructs a rich semantic query string from the user's emotional state."""
        polarity_type = "positive" if positive_polarity > negative_polarity else "negative"
        query_parts = [
            f"Emotion: {dominant_emotion.lower()}.",
            f"Intensity: {emotional_intensity:.2f} ({final_emotional_state}).",
            f"Polarity: {polarity_type}.",
            f"Emotional Context: {final_emotional_state}.",
        ]
        if raw_text:
            query_parts.append(f"User Statement: {raw_text[:120]}.")
        return " ".join(query_parts)

    def match_content(
        self,
        dominant_emotion: str,
        emotional_intensity: float,
        positive_polarity: float,
        negative_polarity: float,
        final_emotional_state: str,
        raw_text: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculates cosine similarity between user emotional query and all wellness content items.
        Returns a dictionary mapping content_id -> {semantic_similarity, matched_emotion, matching_reason}.
        """
        if not self._is_initialized:
            self.initialize_content_embeddings()

        query_text = self.build_emotional_query_text(
            dominant_emotion=dominant_emotion,
            emotional_intensity=emotional_intensity,
            positive_polarity=positive_polarity,
            negative_polarity=negative_polarity,
            final_emotional_state=final_emotional_state,
            raw_text=raw_text,
        )

        query_embedding = self.encode([query_text])[0]  # shape: (384,)
        
        # Cosine similarity: query_emb (normalized) dot product with content_embeddings (normalized)
        cosine_scores = np.dot(self._content_embeddings, query_embedding)
        # Rescale [-1, 1] to [0, 1]
        normalized_scores = np.clip((cosine_scores + 1.0) / 2.0, 0.0, 1.0)

        results: Dict[str, Dict[str, Any]] = {}
        for cid, score in zip(self._content_ids, normalized_scores):
            score_float = float(round(score, 4))
            results[cid] = {
                "content_id": cid,
                "semantic_similarity": score_float,
                "matched_emotion": dominant_emotion,
                "matching_reason": f"Semantic content similarity of {score_float * 100:.1f}% with emotional state '{final_emotional_state}'.",
            }

        return results


# Module singleton
_MATCHER_INSTANCE: Optional[SemanticMatcher] = None


def get_semantic_matcher() -> SemanticMatcher:
    """Returns singleton instance of SemanticMatcher."""
    global _MATCHER_INSTANCE
    if _MATCHER_INSTANCE is None:
        _MATCHER_INSTANCE = SemanticMatcher()
        _MATCHER_INSTANCE.initialize_content_embeddings()
    return _MATCHER_INSTANCE
