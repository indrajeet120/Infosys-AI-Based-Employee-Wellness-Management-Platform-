"""
Validation script for evaluating models on the held-out ISEAR benchmark dataset.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.isear_validation import validate_on_isear_benchmark


def main():
    print("Starting ISEAR benchmark validation...")
    validate_on_isear_benchmark(classifier_or_type="bert")


if __name__ == "__main__":
    main()
