"""Tests for get_data module."""

from unittest.mock import patch

import pytest

from core.get_data import get_data


@pytest.mark.unit
@pytest.mark.data
class TestGetDataInterface:
    """Test high-level get_data interface."""

    @patch("get_data.generate_synthetic_data")
    def test_get_synthetic_data(self, mock_generate):
        """Test getting synthetic data through interface."""
        # Setup mock return
        mock_data = {"dataset": "synthetic", "X_list": []}
        mock_generate.return_value = mock_data

        # Call interface
        result = get_data("synthetic")

        # Verify
        mock_generate.assert_called_once_with(
            num_sources=3, K=3, percW=33.0  # default  # default  # default
        )
        assert result == mock_data

    @patch("get_data.generate_synthetic_data")
    def test_get_synthetic_data_with_params(self, mock_generate):
        """Test synthetic data with custom parameters."""
        mock_data = {"dataset": "synthetic"}
        mock_generate.return_value = mock_data

        result = get_data("synthetic", num_sources=5, K=10, percW=50.0)

        mock_generate.assert_called_once_with(num_sources=5, K=10, percW=50.0)

    def test_get_synthetic_data_aliases(self):
        """Test that synthetic data aliases work."""
        with patch("get_data.generate_synthetic_data") as mock_generate:
            mock_generate.return_value = {"dataset": "synthetic"}

            # Test various aliases
            for alias in ["synthetic", "SYNTHETIC", "toy", "TOY"]:
                get_data(alias)

            # Should be called for each alias
            assert mock_generate.call_count == 4

    @patch("get_data.qmap_pd")
    def test_get_qmap_data(self, mock_qmap_pd):
        """Test getting qMAP-PD data."""
        mock_data = {"dataset": "qmap_pd", "X_list": []}
        mock_qmap_pd.return_value = mock_data

        result = get_data("qmap_pd", data_dir="/fake/dir")

        mock_qmap_pd.assert_called_once_with("/fake/dir")
        assert result == mock_data

    @patch("get_data.qmap_pd")
    def test_get_qmap_data_with_kwargs(self, mock_qmap_pd):
        """Test qMAP-PD data with additional parameters."""
        mock_qmap_pd.return_value = {"dataset": "qmap_pd"}

        get_data(
            "qmap_pd",
            data_dir="/fake/dir",
            clinical_rel="clinical.csv",
            volumes_rel="volumes.csv",
        )

        mock_qmap_pd.assert_called_once_with(
            "/fake/dir", clinical_rel="clinical.csv", volumes_rel="volumes.csv"
        )

    def test_get_qmap_data_aliases(self):
        """Test qMAP-PD data aliases."""
        with patch("get_data.qmap_pd") as mock_qmap_pd:
            mock_qmap_pd.return_value = {"dataset": "qmap_pd"}

            for alias in ["qmap_pd", "QMAP_PD", "qmap-pd", "qmap"]:
                get_data(alias, data_dir="/fake/dir")

            assert mock_qmap_pd.call_count == 4

    def test_qmap_data_missing_data_dir(self):
        """Test that missing data_dir raises error for qMAP-PD."""
        with pytest.raises(ValueError, match="data_dir must be provided"):
            get_data("qmap_pd")

    def test_unknown_dataset_error(self):
        """Test error for unknown dataset."""
        with pytest.raises(ValueError, match="Unknown dataset: unknown"):
            get_data("unknown")

    def test_case_insensitive_dataset_names(self):
        """Test that dataset names are case insensitive."""
        with patch("get_data.generate_synthetic_data") as mock_synth, patch(
            "get_data.qmap_pd"
        ) as mock_qmap:

            mock_synth.return_value = {"dataset": "synthetic"}
            mock_qmap.return_value = {"dataset": "qmap_pd"}

            # Test uppercase
            get_data("SYNTHETIC")
            get_data("QMAP_PD", data_dir="/fake")

            # Test mixed case
            get_data("Synthetic")
            get_data("QMap_Pd", data_dir="/fake")

            # Should have been called
            assert mock_synth.call_count == 2
            assert mock_qmap.call_count == 2
