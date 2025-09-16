"""Tests for visualization manager."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from data import generate_synthetic_data
from visualization.manager import VisualizationManager


class TestVisualizationManager:
    """Test visualization manager functionality."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data for testing."""
        return generate_synthetic_data(num_sources=2, K=3, num_subjects=20, seed=42)

    @pytest.fixture
    def mock_experiment_results(self, synthetic_data):
        """Create mock experiment results."""
        X_list = synthetic_data["X_list"]
        n_subjects = X_list[0].shape[0]
        K = 3

        return {
            "experiment_id": "test_experiment_001",
            "model_results": {
                "factor_scores": np.random.randn(n_subjects, K),
                "factor_loadings": [
                    np.random.randn(X_list[0].shape[1], K),
                    np.random.randn(X_list[1].shape[1], K),
                ],
                "explained_variance": np.array([0.4, 0.3, 0.2]),
                "reconstruction_error": 0.15,
                "convergence_info": {"r_hat": np.array([1.01, 1.02, 1.01])},
            },
            "preprocessing_info": {
                "strategy": "standard",
                "n_features_before": [50, 40],
                "n_features_after": [45, 35],
            },
        }

    @pytest.fixture
    def manager(self):
        """Create VisualizationManager instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            return VisualizationManager(output_dir=Path(tmpdir))

    def test_manager_initialization(self):
        """Test that VisualizationManager initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = VisualizationManager(output_dir=Path(tmpdir))
            assert manager is not None
            assert manager.output_dir.exists()

    def test_generate_experiment_report(self, manager, mock_experiment_results):
        """Test generating complete experiment report."""
        # Should complete without error
        report_path = manager.generate_experiment_report(
            experiment_results=mock_experiment_results,
            experiment_name="test_experiment",
        )

        assert report_path is not None
        assert report_path.exists()

    def test_generate_factor_plots(self, manager, mock_experiment_results):
        """Test generating factor plots."""
        model_results = mock_experiment_results["model_results"]

        # Should complete without error
        plot_paths = manager.generate_factor_plots(
            model_results=model_results, experiment_name="test_experiment"
        )

        assert isinstance(plot_paths, dict)
        assert len(plot_paths) > 0

        # Check that files were created
        for plot_type, path in plot_paths.items():
            assert path.exists(), f"Plot {plot_type} was not created"

    def test_generate_preprocessing_plots(
        self, manager, mock_experiment_results, synthetic_data
    ):
        """Test generating preprocessing plots."""
        X_list = synthetic_data["X_list"]
        preprocessing_info = mock_experiment_results["preprocessing_info"]

        # Should complete without error
        plot_paths = manager.generate_preprocessing_plots(
            X_original=X_list,
            X_processed=X_list,  # Use same data for simplicity
            preprocessing_info=preprocessing_info,
            experiment_name="test_experiment",
        )

        assert isinstance(plot_paths, dict)
        assert len(plot_paths) > 0

        # Check that files were created
        for plot_type, path in plot_paths.items():
            assert path.exists(), f"Preprocessing plot {plot_type} was not created"

    def test_generate_comparison_plots(self, manager, mock_experiment_results):
        """Test generating comparison plots between experiments."""
        # Create results for two experiments
        results_1 = mock_experiment_results
        results_2 = mock_experiment_results.copy()
        results_2["experiment_id"] = "test_experiment_002"

        # Should complete without error
        plot_paths = manager.generate_comparison_plots(
            results_1=results_1,
            results_2=results_2,
            comparison_name="method_comparison",
        )

        assert isinstance(plot_paths, dict)
        assert len(plot_paths) > 0

        # Check that files were created
        for plot_type, path in plot_paths.items():
            assert path.exists(), f"Comparison plot {plot_type} was not created"

    def test_generate_summary_dashboard(self, manager, mock_experiment_results):
        """Test generating summary dashboard."""
        # Create multiple experiment results
        experiment_results = [
            mock_experiment_results,
            {**mock_experiment_results, "experiment_id": "test_experiment_002"},
            {**mock_experiment_results, "experiment_id": "test_experiment_003"},
        ]

        # Should complete without error
        dashboard_path = manager.generate_summary_dashboard(
            experiment_results=experiment_results, dashboard_name="complete_analysis"
        )

        assert dashboard_path is not None
        assert dashboard_path.exists()

    def test_create_html_report(self, manager, mock_experiment_results):
        """Test creating HTML report."""
        # Should complete without error
        html_path = manager.create_html_report(
            experiment_results=mock_experiment_results, report_name="test_report"
        )

        assert html_path is not None
        assert html_path.exists()
        assert html_path.suffix == ".html"

        # Check that HTML content is valid
        with open(html_path, "r") as f:
            content = f.read()
            assert "<html>" in content
            assert "</html>" in content

    def test_save_interactive_plots(self, manager, mock_experiment_results):
        """Test saving interactive plots."""
        model_results = mock_experiment_results["model_results"]

        try:
            # Should complete without error (if interactive plotting available)
            plot_paths = manager.save_interactive_plots(
                model_results=model_results, experiment_name="test_interactive"
            )

            assert isinstance(plot_paths, dict)

            # Check that files were created
            for plot_type, path in plot_paths.items():
                assert path.exists(), f"Interactive plot {plot_type} was not created"

        except (ImportError, NotImplementedError):
            # Interactive plotting might require additional dependencies
            pytest.skip("Interactive plotting not available")

    def test_batch_plot_generation(self, manager):
        """Test batch generation of plots for multiple experiments."""
        # Create multiple mock results
        experiment_results = []
        for i in range(3):
            mock_data = generate_synthetic_data(
                num_sources=2, K=3, num_subjects=15, seed=42 + i
            )
            X_list = mock_data["X_list"]
            n_subjects = X_list[0].shape[0]

            result = {
                "experiment_id": f"batch_experiment_{i:03d}",
                "model_results": {
                    "factor_scores": np.random.randn(n_subjects, 3),
                    "factor_loadings": [
                        np.random.randn(X_list[0].shape[1], 3),
                        np.random.randn(X_list[1].shape[1], 3),
                    ],
                    "explained_variance": np.random.dirichlet([1, 1, 1]),
                    "reconstruction_error": np.random.uniform(0.1, 0.3),
                },
            }
            experiment_results.append(result)

        # Should complete without error
        batch_results = manager.batch_generate_plots(
            experiment_results=experiment_results,
            plot_types=["factor_plots", "summary_plots"],
        )

        assert isinstance(batch_results, dict)
        assert len(batch_results) == len(experiment_results)

    def test_plot_customization(self, manager, mock_experiment_results):
        """Test plot customization options."""
        model_results = mock_experiment_results["model_results"]

        # Test with custom styling
        plot_paths = manager.generate_factor_plots(
            model_results=model_results,
            experiment_name="custom_style",
            style_config={
                "figsize": (12, 8),
                "dpi": 150,
                "colormap": "viridis",
                "style": "seaborn",
            },
        )

        assert isinstance(plot_paths, dict)
        assert len(plot_paths) > 0

        # Check that files were created
        for plot_type, path in plot_paths.items():
            assert path.exists()

    def test_error_handling_invalid_results(self, manager):
        """Test error handling for invalid experiment results."""
        # Test with empty results
        empty_results = {}

        try:
            manager.generate_experiment_report(
                experiment_results=empty_results, experiment_name="invalid_test"
            )
        except (ValueError, KeyError, AttributeError):
            # Expected for invalid results
            pass

        # Test with malformed results
        malformed_results = {
            "experiment_id": "test",
            "model_results": {"factor_scores": "not_an_array"},  # Invalid type
        }

        try:
            manager.generate_factor_plots(
                model_results=malformed_results["model_results"],
                experiment_name="malformed_test",
            )
        except (ValueError, TypeError, AttributeError):
            # Expected for malformed data
            pass

    def test_output_directory_management(self):
        """Test output directory management."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            # Should create nested directories
            manager = VisualizationManager(
                output_dir=base_dir / "deep" / "nested" / "path"
            )
            assert manager.output_dir.exists()

            # Should handle existing directories
            manager2 = VisualizationManager(
                output_dir=base_dir / "deep" / "nested" / "path"
            )
            assert manager2.output_dir.exists()

    def test_cleanup_old_plots(self, manager, mock_experiment_results):
        """Test cleanup of old plot files."""
        model_results = mock_experiment_results["model_results"]

        # Generate plots
        plot_paths_1 = manager.generate_factor_plots(
            model_results=model_results, experiment_name="cleanup_test"
        )

        # Verify files exist
        for path in plot_paths_1.values():
            assert path.exists()

        # Generate plots again (should overwrite)
        plot_paths_2 = manager.generate_factor_plots(
            model_results=model_results, experiment_name="cleanup_test"
        )

        # Files should still exist
        for path in plot_paths_2.values():
            assert path.exists()

    def test_plot_format_options(self, manager, mock_experiment_results):
        """Test different plot format options."""
        model_results = mock_experiment_results["model_results"]

        formats = ["png", "pdf", "svg"]

        for fmt in formats:
            try:
                plot_paths = manager.generate_factor_plots(
                    model_results=model_results,
                    experiment_name=f"format_test_{fmt}",
                    format=fmt,
                )

                # Check that files have correct format
                for path in plot_paths.values():
                    assert path.suffix.lower() == f".{fmt.lower()}"
                    assert path.exists()

            except Exception as e:
                # Some formats might not be available in all environments
                if "format" not in str(e).lower():
                    raise

    def test_memory_efficiency(self, manager):
        """Test memory efficiency with large datasets."""
        # Create larger mock results
        large_results = {
            "experiment_id": "large_test",
            "model_results": {
                "factor_scores": np.random.randn(500, 10),
                "factor_loadings": [
                    np.random.randn(1000, 10),
                    np.random.randn(800, 10),
                ],
                "explained_variance": np.random.dirichlet([1] * 10),
                "reconstruction_error": 0.2,
            },
        }

        # Should handle large data without memory issues
        plot_paths = manager.generate_factor_plots(
            model_results=large_results["model_results"], experiment_name="large_test"
        )

        assert isinstance(plot_paths, dict)
        assert len(plot_paths) > 0

        # Check that files were created
        for path in plot_paths.values():
            assert path.exists()

    def test_concurrent_plot_generation(self, manager, mock_experiment_results):
        """Test concurrent plot generation."""
        import threading

        results = []

        def generate_plots(experiment_id):
            try:
                model_results = mock_experiment_results["model_results"]
                plot_paths = manager.generate_factor_plots(
                    model_results=model_results,
                    experiment_name=f"concurrent_{experiment_id}",
                )
                results.append(len(plot_paths) > 0)
            except Exception:
                results.append(False)

        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=generate_plots, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should succeed
        assert all(results)
