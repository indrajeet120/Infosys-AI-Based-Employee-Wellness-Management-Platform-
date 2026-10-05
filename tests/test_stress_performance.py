"""
Model Performance & Stress Testing Suite (Milestone 4 Task 7).
Evaluates pipeline throughput, latency, CPU memory stability,
and benchmark workloads across Small (10), Medium (50), and Large (200) sample sizes.
"""

import pytest
from services.model_stress_testing import ModelPerformanceStressTester, BenchmarkConfig


@pytest.fixture
def tester():
    return ModelPerformanceStressTester(model_type="distilbert")


def test_stress_test_small_workload(tester):
    config = BenchmarkConfig(sample_count=10, top_k=3, simulate_concurrency=1)
    report = tester.run_benchmark(config)

    assert report.sample_count == 10
    assert report.total_latency_sec > 0.0
    assert report.throughput_qps > 0.0
    assert report.avg_latency_ms > 0.0
    assert report.error_count == 0
    assert report.memory_delta_mb >= 0.0


def test_stress_test_medium_workload(tester):
    config = BenchmarkConfig(sample_count=50, top_k=3, simulate_concurrency=1)
    report = tester.run_benchmark(config)

    assert report.sample_count == 50
    assert report.throughput_qps > 0.0
    assert report.p95_latency_ms >= report.median_latency_ms
    assert report.error_count == 0


def test_stress_test_report_dict_conversion(tester):
    config = BenchmarkConfig(sample_count=10, top_k=2)
    report = tester.run_benchmark(config)
    d = report.to_dict()

    assert "model_type" in d
    assert "throughput_qps" in d
    assert "avg_latency_ms" in d
    assert d["sample_count"] == 10
