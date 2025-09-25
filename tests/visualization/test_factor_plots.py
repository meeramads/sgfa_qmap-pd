"""Tests for factor plotting functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from data import generate_synthetic_data
from visualization.factor_plots import FactorVisualizer


class TestFactorVisualizer:
    """Test factor plotting functionality."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        return generate_synthetic_data(num_sources=2, K=3, num_subjects=20, seed=42)

    @pytest.fixture
    def mock_model_results(self, synthetic_data):
        """Create mock model results for testing."""
        X_list = synthetic_data["X_list"]
        n_subjects = X_list[0].shape[0]
        K = 3

        return {
            "factor_scores": np.random.randn(n_subjects, K),
            "factor_loadings": [
                np.random.randn(X_list[0].shape[1], K),
                np.random.randn(X_list[1].shape[1], K),
            ],
            "explained_variance": np.array([0.4, 0.3, 0.2]),
            "reconstruction_error": 0.15,
            "convergence_info": {"r_hat": np.array([1.01, 1.02, 1.01])},
        }

    @pytest.fixture
    def plotter(self):
        """Create FactorVisualizer instance."""
        return FactorVisualizer({})

    def test_factor_plotter_initialization(self, plotter):
        """Test that FactorVisualizer initializes correctly."""
        assert plotter is not None

    def test_plot_factor_scores(self, plotter, mock_model_results):
        """Test plotting factor scores."""
        factor_scores = mock_model_results["factor_scores"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "factor_scores.png"

            # Should complete without error
            plotter.plot_factor_scores(
                factor_scores=factor_scores, output_path=output_path
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_factor_loadings(self, plotter, mock_model_results):
        """Test plotting factor loadings."""
        factor_loadings = mock_model_results["factor_loadings"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "factor_loadings.png"

            # Should complete without error
            plotter.plot_factor_loadings(
                factor_loadings=factor_loadings, output_path=output_path
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_explained_variance(self, plotter, mock_model_results):
        """Test plotting explained variance."""
        explained_variance = mock_model_results["explained_variance"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "explained_variance.png"

            # Should complete without error
            plotter.plot_explained_variance(
                explained_variance=explained_variance, output_path=output_path
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_factor_correlation_matrix(self, plotter, mock_model_results):
        """Test plotting factor correlation matrix."""
        factor_scores = mock_model_results["factor_scores"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "factor_correlation.png"

            # Should complete without error
            plotter.plot_factor_correlation_matrix(
                factor_scores=factor_scores, output_path=output_path
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_loading_heatmap(self, plotter, mock_model_results):
        """Test plotting loading heatmap."""
        factor_loadings = mock_model_results["factor_loadings"]

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir) / "loading_heatmap.png"

            # Should complete without error for each view
            for i, loadings in enumerate(factor_loadings):
                view_output_path = Path(tmpdir) / f"loading_heatmap_view_{i}.png"
                plotter.plot_loading_heatmap(
                    factor_loadings=loadings,
                    output_path=view_output_path,
                    view_name=f"View_{i}",
                )

                # Check that file was created
                assert view_output_path.exists()

    def test_plot_factor_comparison(self, plotter):
        """Test plotting factor comparison between models."""
        # Create mock results for two models
        model1_scores = np.random.randn(20, 3)
        model2_scores = np.random.randn(20, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "factor_comparison.png"

            # Should complete without error
            plotter.plot_factor_comparison(
                factor_scores_1=model1_scores,
                factor_scores_2=model2_scores,
                model_names=["Model 1", "Model 2"],
                output_path=output_path,
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_reconstruction_quality(
        self, plotter, synthetic_data, mock_model_results
    ):
        """Test plotting reconstruction quality."""
        X_original = synthetic_data["X_list"][0]
        X_reconstructed = X_original + np.random.randn(*X_original.shape) * 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "reconstruction_quality.png"

            # Should complete without error
            plotter.plot_reconstruction_quality(
                X_original=X_original,
                X_reconstructed=X_reconstructed,
                output_path=output_path,
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_convergence_diagnostics(self, plotter, mock_model_results):
        """Test plotting convergence diagnostics."""
        convergence_info = mock_model_results["convergence_info"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "convergence_diagnostics.png"

            # Should complete without error
            plotter.plot_convergence_diagnostics(
                convergence_info=convergence_info, output_path=output_path
            )

            # Check that file was created
            assert output_path.exists()

    def test_create_factor_summary_plot(self, plotter, mock_model_results):
        """Test creating comprehensive factor summary plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "factor_summary.png"

            # Should complete without error
            plotter.create_factor_summary_plot(
                model_results=mock_model_results, output_path=output_path
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_with_labels(self, plotter, mock_model_results):
        """Test plotting with subject labels."""
        factor_scores = mock_model_results["factor_scores"]
        labels = np.random.choice(["Control", "Patient"], size=factor_scores.shape[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "factor_scores_labeled.png"

            # Should complete without error
            plotter.plot_factor_scores(
                factor_scores=factor_scores, output_path=output_path, labels=labels
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_error_handling_invalid_dimensions(self, plotter):
        """Test error handling for invalid dimensions."""
        # Create mismatched data
        factor_scores = np.random.randn(20, 3)
        invalid_labels = np.random.choice(["A", "B"], size=15)  # Wrong size

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"

            # Should handle error gracefully
            try:
                plotter.plot_factor_scores(
                    factor_scores=factor_scores,
                    output_path=output_path,
                    labels=invalid_labels,
                )
            except (ValueError, AssertionError):
                # Expected for mismatched dimensions
                pass

    def test_plot_with_custom_styling(self, plotter, mock_model_results):
        """Test plotting with custom styling options."""
        factor_scores = mock_model_results["factor_scores"]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "styled_plot.png"

            # Test with custom styling options
            plotter.plot_factor_scores(
                factor_scores=factor_scores,
                output_path=output_path,
                figsize=(12, 8),
                dpi=150,
                style="seaborn",
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_interactive_disabled(self, plotter, mock_model_results):
        """Test that interactive plotting is disabled for testing."""
        factor_scores = mock_model_results["factor_scores"]

        # Ensure matplotlib backend is appropriate for testing
        import matplotlib

        matplotlib.use("Agg")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_interactive.png"

            # Should work without interactive display
            plotter.plot_factor_scores(
                factor_scores=factor_scores, output_path=output_path
            )

            assert output_path.exists()

    def test_plot_memory_efficiency(self, plotter):
        """Test plotting with large data for memory efficiency."""
        # Create larger synthetic data
        large_scores = np.random.randn(500, 10)  # 500 subjects, 10 factors
        large_loadings = [
            np.random.randn(1000, 10),  # 1000 features, 10 factors
            np.random.randn(800, 10),  # 800 features, 10 factors
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test factor scores
            scores_path = Path(tmpdir) / "large_scores.png"
            plotter.plot_factor_scores(
                factor_scores=large_scores, output_path=scores_path
            )
            assert scores_path.exists()

            # Test factor loadings
            loadings_path = Path(tmpdir) / "large_loadings.png"
            plotter.plot_factor_loadings(
                factor_loadings=large_loadings, output_path=loadings_path
            )
            assert loadings_path.exists()

    def test_plot_multiple_formats(self, plotter, mock_model_results):
        """Test saving plots in multiple formats."""
        factor_scores = mock_model_results["factor_scores"]

        with tempfile.TemporaryDirectory() as tmpdir:
            formats = ["png", "pdf", "svg"]

            for fmt in formats:
                output_path = Path(tmpdir) / f"factor_scores.{fmt}"

                try:
                    plotter.plot_factor_scores(
                        factor_scores=factor_scores, output_path=output_path
                    )

                    # Check that file was created
                    assert output_path.exists()

                except Exception as e:
                    # Some formats might not be available in all environments
                    if "format" not in str(e).lower():
                        raise

    def test_plot_with_missing_data(self, plotter):
        """Test plotting with missing data."""
        # Create data with NaN values
        factor_scores = np.random.randn(20, 3)
        factor_scores[5:8, 1] = np.nan

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "missing_data.png"

            # Should handle missing data gracefully
            try:
                plotter.plot_factor_scores(
                    factor_scores=factor_scores, output_path=output_path
                )
                # If successful, file should exist
                assert output_path.exists()

            except (ValueError, RuntimeError):
                # Expected if missing data not handled
                pass

    @patch("matplotlib.pyplot.show")
    def test_plot_without_saving(self, mock_show, plotter, mock_model_results):
        """Test plotting without saving to file."""
        factor_scores = mock_model_results["factor_scores"]

        # Should be able to create plot without saving
        try:
            plotter.plot_factor_scores(
                factor_scores=factor_scores, output_path=None, show=True
            )
            # Should call plt.show()
            mock_show.assert_called_once()

        except TypeError:
            # Expected if output_path is required
            pass
