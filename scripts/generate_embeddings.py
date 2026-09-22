"""
Precomputes and persists wellness content embeddings for semantic matching.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.semantic_matcher import SemanticMatcher
from services.wellness_data import get_wellness_repository


def main():
    print("=" * 70)
    print("GENERATING & CACHING WELLNESS CONTENT EMBEDDINGS")
    print("=" * 70)
    
    repo = get_wellness_repository()
    print(f"Loaded {repo.count()} wellness activities from dataset.")

    matcher = SemanticMatcher()
    matcher.initialize_content_embeddings(force_recompute=True)
    
    print(f"Successfully generated embeddings matrix with shape: {matcher._content_embeddings.shape}")
    print("Embeddings saved to models/wellness_embeddings.npy")
    print("=" * 70)


if __name__ == "__main__":
    main()
