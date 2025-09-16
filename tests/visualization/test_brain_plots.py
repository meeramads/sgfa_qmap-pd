"""Tests for brain plotting functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from data import generate_synthetic_data
from visualization.brain_plots import BrainPlotter


class TestBrainPlotter:
    """Test brain plotting functionality."""

    @pytest.fixture
    def mock_brain_data(self):
        """Create mock brain data for testing."""
        return {
            "factor_loadings": np.random.randn(100, 3),  # 100 brain regions, 3 factors
            "factor_scores": np.random.randn(20, 3),  # 20 subjects, 3 factors
            "brain_coordinates": {
                "x": np.random.uniform(-50, 50, 100),
                "y": np.random.uniform(-50, 50, 100),
                "z": np.random.uniform(-50, 50, 100),
            },
            "region_names": [f"Region_{i}" for i in range(100)],
            "hemisphere": np.random.choice(["L", "R"], 100),
        }

    @pytest.fixture
    def plotter(self):
        """Create BrainPlotter instance."""
        return BrainPlotter()

    def test_brain_plotter_initialization(self, plotter):
        """Test that BrainPlotter initializes correctly."""
        assert plotter is not None

    def test_plot_brain_factor_loadings(self, plotter, mock_brain_data):
        """Test plotting brain factor loadings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "brain_loadings.png"

            # Should complete without error
            plotter.plot_brain_factor_loadings(
                factor_loadings=mock_brain_data["factor_loadings"],
                coordinates=mock_brain_data["brain_coordinates"],
                output_path=output_path,
                factor_index=0,
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_glass_brain(self, plotter, mock_brain_data):
        """Test glass brain plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "glass_brain.png"

            # Should complete without error
            plotter.plot_glass_brain(
                brain_map=mock_brain_data["factor_loadings"][:, 0],
                coordinates=mock_brain_data["brain_coordinates"],
                output_path=output_path,
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_surface_brain(self, plotter, mock_brain_data):
        """Test surface brain plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "surface_brain.png"

            try:
                # Should complete without error (if surface plotting is available)
                plotter.plot_surface_brain(
                    brain_map=mock_brain_data["factor_loadings"][:, 0],
                    output_path=output_path,
                )

                # Check that file was created
                assert output_path.exists()

            except (ImportError, NotImplementedError):
                # Surface plotting might require additional dependencies
                pytest.skip("Surface plotting not available")

    def test_plot_roi_loadings(self, plotter, mock_brain_data):
        """Test ROI loadings plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "roi_loadings.png"

            # Should complete without error
            plotter.plot_roi_loadings(
                factor_loadings=mock_brain_data["factor_loadings"],
                region_names=mock_brain_data["region_names"],
                output_path=output_path,
                factor_index=0,
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_hemisphere_comparison(self, plotter, mock_brain_data):
        """Test hemisphere comparison plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "hemisphere_comparison.png"

            # Should complete without error
            plotter.plot_hemisphere_comparison(
                factor_loadings=mock_brain_data["factor_loadings"][:, 0],
                hemisphere_labels=mock_brain_data["hemisphere"],
                output_path=output_path,
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_connectivity_matrix(self, plotter, mock_brain_data):
        """Test connectivity matrix plotting."""
        # Create mock connectivity matrix
        n_regions = len(mock_brain_data["region_names"])
        connectivity_matrix = np.random.randn(n_regions, n_regions)
        # Make symmetric
        connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "connectivity_matrix.png"

            # Should complete without error
            plotter.plot_connectivity_matrix(
                connectivity_matrix=connectivity_matrix,
                region_names=mock_brain_data["region_names"],
                output_path=output_path,
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_factor_brain_maps(self, plotter, mock_brain_data):
        """Test plotting multiple factor brain maps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "factor_brain_maps.png"

            # Should complete without error
            plotter.plot_factor_brain_maps(
                factor_loadings=mock_brain_data["factor_loadings"],
                coordinates=mock_brain_data["brain_coordinates"],
                output_path=output_path,
                n_factors=3,
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_thresholded_brain_map(self, plotter, mock_brain_data):
        """Test plotting thresholded brain maps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "thresholded_brain.png"

            # Should complete without error
            plotter.plot_thresholded_brain_map(
                brain_map=mock_brain_data["factor_loadings"][:, 0],
                coordinates=mock_brain_data["brain_coordinates"],
                threshold=0.5,
                output_path=output_path,
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_brain_networks(self, plotter, mock_brain_data):
        """Test plotting brain networks."""
        # Create mock network connectivity
        n_regions = len(mock_brain_data["region_names"])
        network_matrix = np.random.rand(n_regions, n_regions)
        network_matrix[network_matrix < 0.8] = 0  # Sparse connections

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "brain_networks.png"

            try:
                # Should complete without error
                plotter.plot_brain_networks(
                    connectivity_matrix=network_matrix,
                    coordinates=mock_brain_data["brain_coordinates"],
                    output_path=output_path,
                )

                # Check that file was created
                assert output_path.exists()

            except (ImportError, NotImplementedError):
                # Network plotting might require additional dependencies
                pytest.skip("Brain network plotting not available")

    def test_plot_with_atlas_labels(self, plotter, mock_brain_data):
        """Test plotting with atlas labels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "atlas_brain.png"

            # Create mock atlas labels
            atlas_labels = np.random.choice(
                ["Frontal", "Parietal", "Temporal", "Occipital"],
                len(mock_brain_data["region_names"]),
            )

            # Should complete without error
            plotter.plot_brain_factor_loadings(
                factor_loadings=mock_brain_data["factor_loadings"],
                coordinates=mock_brain_data["brain_coordinates"],
                output_path=output_path,
                factor_index=0,
                atlas_labels=atlas_labels,
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_statistical_map(self, plotter, mock_brain_data):
        """Test plotting statistical brain maps."""
        # Create mock statistical values (z-scores)
        z_scores = np.random.randn(len(mock_brain_data["region_names"]))
        p_values = 2 * (1 - np.random.random(len(mock_brain_data["region_names"])))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "statistical_brain.png"

            # Should complete without error
            plotter.plot_statistical_brain_map(
                z_scores=z_scores,
                p_values=p_values,
                coordinates=mock_brain_data["brain_coordinates"],
                output_path=output_path,
                alpha=0.05,
            )

            # Check that file was created
            assert output_path.exists()

    def test_plot_error_handling_mismatched_dimensions(self, plotter):
        """Test error handling for mismatched dimensions."""
        # Create mismatched data
        factor_loadings = np.random.randn(100, 3)
        coordinates = {
            "x": np.random.randn(50),  # Wrong size
            "y": np.random.randn(50),
            "z": np.random.randn(50),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"

            # Should handle error gracefully
            with pytest.raises((ValueError, AssertionError, IndexError)):
                plotter.plot_brain_factor_loadings(
                    factor_loadings=factor_loadings,
                    coordinates=coordinates,
                    output_path=output_path,
                    factor_index=0,
                )

    def test_plot_with_missing_coordinates(self, plotter, mock_brain_data):
        """Test plotting with missing coordinate information."""
        incomplete_coordinates = {
            "x": mock_brain_data["brain_coordinates"]["x"],
            "y": mock_brain_data["brain_coordinates"]["y"],
            # Missing z coordinate
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "missing_coords.png"

            # Should handle missing coordinates gracefully
            try:
                plotter.plot_brain_factor_loadings(
                    factor_loadings=mock_brain_data["factor_loadings"],
                    coordinates=incomplete_coordinates,
                    output_path=output_path,
                    factor_index=0,
                )
            except (KeyError, ValueError):
                # Expected for missing coordinates
                pass

    def test_plot_different_colormaps(self, plotter, mock_brain_data):
        """Test plotting with different colormaps."""
        colormaps = ["viridis", "coolwarm", "RdBu_r", "plasma"]

        with tempfile.TemporaryDirectory() as tmpdir:
            for cmap in colormaps:
                output_path = Path(tmpdir) / f"brain_{cmap}.png"

                try:
                    plotter.plot_brain_factor_loadings(
                        factor_loadings=mock_brain_data["factor_loadings"],
                        coordinates=mock_brain_data["brain_coordinates"],
                        output_path=output_path,
                        factor_index=0,
                        colormap=cmap,
                    )

                    # Check that file was created
                    assert output_path.exists()

                except Exception as e:
                    # Some colormaps might not be available
                    if "colormap" not in str(e).lower():
                        raise

    def test_plot_multiple_views(self, plotter, mock_brain_data):
        """Test plotting multiple brain views."""
        views = ["lateral", "medial", "dorsal", "ventral"]

        with tempfile.TemporaryDirectory() as tmpdir:
            for view in views:
                output_path = Path(tmpdir) / f"brain_{view}.png"

                try:
                    plotter.plot_brain_factor_loadings(
                        factor_loadings=mock_brain_data["factor_loadings"],
                        coordinates=mock_brain_data["brain_coordinates"],
                        output_path=output_path,
                        factor_index=0,
                        view=view,
                    )

                    # Check that file was created
                    assert output_path.exists()

                except (ValueError, NotImplementedError):
                    # Some views might not be supported
                    continue

    def test_plot_interactive_disabled(self, plotter, mock_brain_data):
        """Test that interactive plotting is disabled for testing."""
        # Ensure matplotlib backend is appropriate for testing
        import matplotlib

        matplotlib.use("Agg")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_interactive.png"

            # Should work without interactive display
            plotter.plot_brain_factor_loadings(
                factor_loadings=mock_brain_data["factor_loadings"],
                coordinates=mock_brain_data["brain_coordinates"],
                output_path=output_path,
                factor_index=0,
            )

            assert output_path.exists()

    def test_plot_memory_efficiency(self, plotter):
        """Test plotting with large brain data."""
        # Create larger brain data
        large_loadings = np.random.randn(1000, 5)  # 1000 regions, 5 factors
        large_coordinates = {
            "x": np.random.uniform(-100, 100, 1000),
            "y": np.random.uniform(-100, 100, 1000),
            "z": np.random.uniform(-100, 100, 1000),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "large_brain.png"

            # Should handle large data without memory issues
            plotter.plot_brain_factor_loadings(
                factor_loadings=large_loadings,
                coordinates=large_coordinates,
                output_path=output_path,
                factor_index=0,
            )

            assert output_path.exists()

    @patch("matplotlib.pyplot.show")
    def test_plot_without_saving(self, mock_show, plotter, mock_brain_data):
        """Test plotting without saving to file."""
        # Should be able to create plot without saving
        try:
            plotter.plot_brain_factor_loadings(
                factor_loadings=mock_brain_data["factor_loadings"],
                coordinates=mock_brain_data["brain_coordinates"],
                output_path=None,
                factor_index=0,
                show=True,
            )
            # Should call plt.show()
            mock_show.assert_called_once()

        except TypeError:
            # Expected if output_path is required
            pass
