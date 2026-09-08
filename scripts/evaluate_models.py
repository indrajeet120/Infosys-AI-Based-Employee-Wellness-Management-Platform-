"""
Evaluation script to compare BERT and DistilBERT models.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.evaluation import compare_models


def main():
    print("Executing model evaluation and comparative analysis...")
    compare_models()


if __name__ == "__main__":
    main()
