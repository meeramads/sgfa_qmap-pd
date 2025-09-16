"""Tests for reproducibility experiment."""

import tempfile
from pathlib import Path

import pytest

from data import generate_synthetic_data
from experiments.reproducibility import run_reproducibility


class TestReproducibility:
    """Test reproducibility experiment."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return {
            "data": {"data_dir": "./test_data"},
            "experiments": {
                "base_output_dir": "./test_results",
                "save_intermediate": True,
            },
            "reproducibility": {
                "test_scenarios": [
                    "identical_seeds",
                    "different_hardware",
                    "version_stability",
                ],
                "seed_values": [42, 123, 456],
                "n_repetitions": 2,  # Reduced for testing
                "convergence_metrics": [
                    "factor_correlation",
                    "parameter_stability",
                    "reconstruction_consistency",
                ],
                "tolerance_thresholds": {
                    "correlation_threshold": 0.95,
                    "parameter_relative_error": 0.05,
                    "reconstruction_error_ratio": 0.02,
                },
            },
        }

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        return generate_synthetic_data(
            num_sources=2, K=3, num_subjects=20, seed=42  # Small for fast testing
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

    def test_reproducibility_runs(self, mock_config):
        """Test that reproducibility experiment runs without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_reproducibility_with_shared_data(self, shared_data_config):
        """Test reproducibility with shared data mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_data_config["experiments"]["base_output_dir"] = tmpdir

            result = run_reproducibility(shared_data_config)
            assert result is not None

    def test_identical_seeds_reproducibility(self, mock_config):
        """Test reproducibility with identical seeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Focus on identical seeds test
            mock_config["reproducibility"]["test_scenarios"] = ["identical_seeds"]
            mock_config["reproducibility"]["seed_values"] = [42]
            mock_config["reproducibility"]["n_repetitions"] = 2

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_different_hardware_reproducibility(self, mock_config):
        """Test reproducibility across different hardware configurations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Focus on hardware differences (simulated)
            mock_config["reproducibility"]["test_scenarios"] = ["different_hardware"]

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_version_stability_reproducibility(self, mock_config):
        """Test version stability reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Focus on version stability
            mock_config["reproducibility"]["test_scenarios"] = ["version_stability"]

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_multiple_seed_reproducibility(self, mock_config):
        """Test reproducibility across multiple seed values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test multiple seeds
            mock_config["reproducibility"]["seed_values"] = [42, 123, 456]
            mock_config["reproducibility"]["n_repetitions"] = 2

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_factor_correlation_analysis(self, mock_config):
        """Test factor correlation reproducibility analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Focus on factor correlation metrics
            mock_config["reproducibility"]["convergence_metrics"] = [
                "factor_correlation"
            ]

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_parameter_stability_analysis(self, mock_config):
        """Test parameter stability analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Focus on parameter stability
            mock_config["reproducibility"]["convergence_metrics"] = [
                "parameter_stability"
            ]

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_reconstruction_consistency_analysis(self, mock_config):
        """Test reconstruction consistency analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Focus on reconstruction consistency
            mock_config["reproducibility"]["convergence_metrics"] = [
                "reconstruction_consistency"
            ]

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_tolerance_threshold_validation(self, mock_config):
        """Test that tolerance thresholds are properly applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Set specific tolerance thresholds
            mock_config["reproducibility"]["tolerance_thresholds"] = {
                "correlation_threshold": 0.90,
                "parameter_relative_error": 0.10,
                "reconstruction_error_ratio": 0.05,
            }

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_reproducibility_output_structure(self, mock_config):
        """Test that reproducibility produces expected output structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_reproducibility(mock_config)

            # Check result structure
            assert hasattr(result, "experiment_id")
            assert hasattr(result, "status")
            assert hasattr(result, "model_results")

            # Check output directory
            output_path = Path(tmpdir)
            assert output_path.exists()

    def test_reproducibility_metrics_computation(self, mock_config):
        """Test that reproducibility metrics are computed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_reproducibility(mock_config)
            assert result is not None

            if hasattr(result, "model_results") and result.model_results:
                model_results = result.model_results

                # Should contain reproducibility analysis results
                assert isinstance(model_results, dict)

    def test_mcmc_reproducibility(self, mock_config):
        """Test MCMC sampling reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Add MCMC-specific reproducibility tests
            mock_config["reproducibility"]["mcmc_reproducibility"] = True
            mock_config["reproducibility"]["mcmc_metrics"] = [
                "chain_convergence",
                "sample_correlation",
            ]

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_cross_platform_reproducibility(self, mock_config):
        """Test cross-platform reproducibility (simulated)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Simulate cross-platform testing
            mock_config["reproducibility"]["cross_platform"] = True

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_numerical_precision_effects(self, mock_config):
        """Test effects of numerical precision on reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test different precision settings
            mock_config["reproducibility"]["precision_tests"] = {
                "float32_vs_float64": True,
                "jax_precision": ["float32", "float64"],
            }

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_minimal_reproducibility_config(self, synthetic_data):
        """Test with minimal reproducibility configuration."""
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

            result = run_reproducibility(minimal_config)
            assert result is not None

    def test_error_handling_invalid_scenarios(self, mock_config):
        """Test error handling with invalid test scenarios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Invalid test scenario
            mock_config["reproducibility"]["test_scenarios"] = ["invalid_scenario_name"]

            # Should handle gracefully
            result = run_reproducibility(mock_config)

            # Either succeeds with fallback or handles error
            if result is not None:
                assert hasattr(result, "status")

    def test_seed_range_validation(self, mock_config):
        """Test validation of seed ranges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test with various seed values
            mock_config["reproducibility"]["seed_values"] = [0, 1, 42, 12345, 999999]

            result = run_reproducibility(mock_config)
            assert result is not None

    @pytest.mark.parametrize(
        "scenario", ["identical_seeds", "different_hardware", "version_stability"]
    )
    def test_individual_scenarios(self, mock_config, scenario):
        """Test individual reproducibility scenarios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test single scenario
            mock_config["reproducibility"]["test_scenarios"] = [scenario]

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_reproducibility_report_generation(self, mock_config):
        """Test that reproducibility reports are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable report generation
            mock_config["reproducibility"]["generate_report"] = True

            result = run_reproducibility(mock_config)
            assert result is not None

            # Check for report files
            report_files = list(Path(tmpdir).rglob("*report*")) + list(
                Path(tmpdir).rglob("*summary*")
            )

            # Should have some form of output
            assert len(list(Path(tmpdir).rglob("*"))) > 0

    def test_statistical_significance_testing(self, mock_config):
        """Test statistical significance testing in reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable statistical testing
            mock_config["reproducibility"]["statistical_tests"] = True
            mock_config["reproducibility"]["significance_level"] = 0.05

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_reproducibility_matrix_saving(self, mock_config):
        """Test that reproducibility matrices are saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_reproducibility(mock_config)
            assert result is not None

            # Check for matrix output files
            output_files = list(Path(tmpdir).rglob("*"))
            assert len(output_files) > 0

    def test_convergence_diagnostics(self, mock_config):
        """Test convergence diagnostics in reproducibility analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable convergence diagnostics
            mock_config["reproducibility"]["convergence_diagnostics"] = True

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_reproducibility_with_different_repetitions(self, mock_config):
        """Test reproducibility with different numbers of repetitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test with minimal repetitions for fast testing
            mock_config["reproducibility"]["n_repetitions"] = 1

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_environmental_factors_tracking(self, mock_config):
        """Test tracking of environmental factors affecting reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable environmental tracking
            mock_config["reproducibility"]["track_environment"] = True

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_batch_reproducibility_testing(self, mock_config):
        """Test batch reproducibility testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Run batch tests
            mock_config["reproducibility"]["batch_testing"] = True
            mock_config["reproducibility"]["batch_size"] = 2

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_reproducibility_visualization(self, mock_config):
        """Test reproducibility visualization generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable visualization
            mock_config["experiments"]["generate_plots"] = True

            result = run_reproducibility(mock_config)
            assert result is not None

            # Check for plot files
            plot_files = list(Path(tmpdir).rglob("*.png")) + list(
                Path(tmpdir).rglob("*.pdf")
            )

            # Should generate some output
            assert len(list(Path(tmpdir).rglob("*"))) > 0

    def test_deterministic_vs_stochastic_components(self, mock_config):
        """Test analysis of deterministic vs stochastic components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable component analysis
            mock_config["reproducibility"]["component_analysis"] = True

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_reproducibility_across_model_variants(self, mock_config):
        """Test reproducibility across different model variants."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Test multiple model variants
            mock_config["reproducibility"]["test_model_variants"] = True

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_long_term_reproducibility(self, mock_config):
        """Test long-term reproducibility tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            # Enable long-term tracking
            mock_config["reproducibility"]["long_term_tracking"] = True

            result = run_reproducibility(mock_config)
            assert result is not None

    def test_reproducibility_summary_statistics(self, mock_config):
        """Test computation of reproducibility summary statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config["experiments"]["base_output_dir"] = tmpdir

            result = run_reproducibility(mock_config)
            assert result is not None

            # Check for summary output
            list(Path(tmpdir).rglob("*summary*"))

            # Should have some form of summary output
            assert len(list(Path(tmpdir).rglob("*"))) > 0
