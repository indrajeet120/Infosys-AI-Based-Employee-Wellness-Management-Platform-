"""
Unit tests for the ISEAR Benchmark Validation service.
"""

from pathlib import Path
import pytest
from services.config import ISEAR_BENCHMARK_PATH, ISEAR_RESULTS_PATH, ISEAR_METRICS_PATH
from services.isear_validation import validate_on_isear_benchmark


class TestIsearValidation:
    def test_isear_dataset_exists_and_valid(self):
        assert ISEAR_BENCHMARK_PATH.exists()

    def test_isear_validation_execution(self):
        result = validate_on_isear_benchmark("bert", isear_data_path=ISEAR_BENCHMARK_PATH)
        assert "metrics" in result
        assert "results_df" in result
        
        metrics = result["metrics"]
        assert metrics["total_samples"] == 23
        assert metrics["sample_accuracy"] > 0.0
        assert metrics["hamming_accuracy"] > 0.0
        assert "emotion_wise_performance" in metrics
        
        # Verify reports are generated
        assert ISEAR_RESULTS_PATH.exists()
        assert ISEAR_METRICS_PATH.exists()
