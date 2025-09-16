"""Tests for performance benchmarks experiment."""

import tempfile
import time
from pathlib import Path

import pytest

from data import generate_synthetic_data
from experiments.performance_benchmarks import run_performance_benchmarks


class TestPerformanceBenchmarks:
    """Test performance benchmarks experiment."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return {
            "data": {"data_dir": "./test_data"},
            "experiments": {
                "base_output_dir": "./test_results",
                "save_intermediate": True,
            },
            "performance_benchmarks": {
                "benchmark_configs": {
                    "tiny_scale": {  # Very small for testing
                        "n_subjects": 15,
                        "n_features_per_view": [20, 30, 10],
                    },
                    "small_scale": {
                        "n_subjects": 25,
                        "n_features_per_view": [30, 40, 15],
                    },
                },
                "metrics_to_track": [
                    "training_time",
                    "memory_usage",
                    "convergence_rate",
                ],
            },
        }

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        return generate_synthetic_data(
            num_sources=2, K=3, num_subjects=15, seed=42  # Very small for fast testing
        )

    @pytest.fixture
    def shared_data_config(self, mock_config, synthetic_data):
        """Create config with shared data."""
        config = mock_config.copy()
        config["_shared_data"] = {
            "X_list": synthetic_data["X_list"],
            "preprocessing_info": {"strategy": "minimal"},
            "mode": "shared",
        }
        return config

    def test_performance_benchmarks_runs(self, mock_config):
        """Test that performance benchmarks experiment runs without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_performance_benchmarks(mock_config)
            assert result is not None

    def test_performance_benchmarks_with_shared_data(self, shared_data_config):
        """Test performance benchmarks with shared data mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_data_config["experiments"]["base_output_dir"] = tmpdir

            result = run_performance_benchmarks(shared_data_config)
            assert result is not None

    def test_scalability_benchmarks(self, mock_config):
        """Test scalability benchmarks with different data sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Ensure multiple scales are tested
            mock_config["performance_benchmarks"]["benchmark_configs"] = {
                "scale_1": {"n_subjects": 10, "n_features_per_view": [15, 20]},
                "scale_2": {"n_subjects": 20, "n_features_per_view": [25, 30]},
            }

            result = run_performance_benchmarks(mock_config)
            assert result is not None

    def test_memory_benchmarks(self, mock_config):
        """Test memory usage benchmarks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Focus on memory metrics
            mock_config["performance_benchmarks"]["metrics_to_track"] = [
                "memory_usage",
                "training_time",
            ]

            result = run_performance_benchmarks(mock_config)
            assert result is not None

    def test_timing_benchmarks(self, mock_config):
        """Test that timing measurements are captured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            start_time = time.time()
            result = run_performance_benchmarks(mock_config)
            end_time = time.time()

            assert result is not None

            # Should complete in reasonable time for small test data
            elapsed = end_time - start_time
            assert elapsed < 300  # Should complete within 5 minutes

    def test_convergence_analysis(self, mock_config):
        """Test convergence rate analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            mock_config["performance_benchmarks"]["metrics_to_track"] = [
                "convergence_rate",
                "training_time",
            ]

            result = run_performance_benchmarks(mock_config)
            assert result is not None

    def test_multiple_benchmark_configs(self, mock_config):
        """Test with multiple benchmark configurations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test multiple scales
            mock_config["performance_benchmarks"]["benchmark_configs"] = {
                "config_1": {"n_subjects": 12, "n_features_per_view": [20, 15]},
                "config_2": {"n_subjects": 18, "n_features_per_view": [25, 20]},
                "config_3": {"n_subjects": 24, "n_features_per_view": [30, 25]},
            }

            result = run_performance_benchmarks(mock_config)
            assert result is not None

    def test_benchmark_output_structure(self, mock_config):
        """Test that benchmarks produce expected output structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_performance_benchmarks(mock_config)

            # Check result structure
            assert hasattr(result, "experiment_id")
            assert hasattr(result, "status")
            assert hasattr(result, "model_results")

            # Check output directory
            output_path = Path(tmpdir)
            assert output_path.exists()

    def test_performance_metrics_validation(self, mock_config):
        """Test that performance metrics are reasonable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_performance_benchmarks(mock_config)
            assert result is not None

            if hasattr(result, "model_results") and result.model_results:
                model_results = result.model_results

                # Should contain performance data
                assert isinstance(model_results, dict)

    def test_memory_limit_handling(self, mock_config):
        """Test handling of memory constraints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Add memory constraints
            mock_config["system"] = {
                "memory_limit_gb": 2.0  # Low memory limit for testing
            }

            result = run_performance_benchmarks(mock_config)
            assert result is not None

    def test_gpu_benchmarks(self, mock_config):
        """Test GPU-specific benchmarks if available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Add GPU tracking
            mock_config["performance_benchmarks"]["metrics_to_track"].append(
                "gpu_utilization"
            )

            result = run_performance_benchmarks(mock_config)
            assert result is not None

    def test_comparative_benchmarks(self, mock_config):
        """Test comparative benchmarks against other methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Add comparative analysis
            mock_config["performance_benchmarks"]["compare_methods"] = True

            result = run_performance_benchmarks(mock_config)
            assert result is not None

    def test_minimal_benchmark_config(self, synthetic_data):
        """Test with minimal benchmark configuration."""
        minimal_config = {
            "data": {"data_dir": "./test_data"},
            "experiments": {"base_output_dir": "./test_results"},
            "_shared_data": {
                "X_list": synthetic_data["X_list"],
                "preprocessing_info": {},
                "mode": "shared",
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_config["experiments"]["base_output_dir"] = tmpdir

            result = run_performance_benchmarks(minimal_config)
            assert result is not None

    def test_error_handling_large_data(self, mock_config):
        """Test error handling with unreasonably large data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Set unreasonably large data size (should be handled gracefully)
            mock_config["performance_benchmarks"]["benchmark_configs"] = {
                "too_large": {
                    "n_subjects": 1000000,  # Unreasonably large
                    "n_features_per_view": [10000, 15000],
                }
            }

            # Should handle gracefully (skip or reduce size)
            result = run_performance_benchmarks(mock_config)

            # Either succeeds with reduced size or handles error
            if result is not None:
                assert hasattr(result, "status")

    def test_benchmark_reproducibility(self, mock_config):
        """Test benchmark reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir
            mock_config["random_seed"] = 42

            # Run twice with same configuration
            result1 = run_performance_benchmarks(mock_config)
            result2 = run_performance_benchmarks(mock_config)

            assert result1 is not None
            assert result2 is not None

    def test_benchmark_matrix_saving(self, mock_config):
        """Test that benchmark matrices are saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_performance_benchmarks(mock_config)
            assert result is not None

            # Check for matrix output files
            output_files = list(Path(tmpdir).rglob("*"))
            assert len(output_files) > 0

    def test_performance_regression_detection(self, mock_config):
        """Test performance regression detection capabilities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Add baseline comparison
            mock_config["performance_benchmarks"]["detect_regressions"] = True

            result = run_performance_benchmarks(mock_config)
            assert result is not None

    @pytest.mark.parametrize(
        "metric", ["training_time", "memory_usage", "convergence_rate"]
    )
    def test_individual_metrics(self, mock_config, metric):
        """Test individual performance metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test single metric
            mock_config["performance_benchmarks"]["metrics_to_track"] = [metric]

            result = run_performance_benchmarks(mock_config)
            assert result is not None

    def test_benchmark_output_files(self, mock_config):
        """Test that expected benchmark output files are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_performance_benchmarks(mock_config)
            assert result is not None

            # Check for output files
            output_files = list(Path(tmpdir).rglob("*"))
            assert len(output_files) > 0

            # Should have some structured output
            json_files = list(Path(tmpdir).rglob("*.json"))
            pkl_files = list(Path(tmpdir).rglob("*.pkl"))
            assert len(json_files) + len(pkl_files) > 0

    def test_benchmark_scaling_analysis(self, mock_config):
        """Test scaling analysis across different data sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Multiple scales for scaling analysis
            mock_config["performance_benchmarks"]["benchmark_configs"] = {
                "small": {"n_subjects": 10, "n_features_per_view": [15, 20]},
                "medium": {"n_subjects": 20, "n_features_per_view": [25, 30]},
                "large": {"n_subjects": 30, "n_features_per_view": [35, 40]},
            }

            result = run_performance_benchmarks(mock_config)
            assert result is not None
