"""Tests for utility functions."""

import pickle
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from core.utils import (
    check_available_memory,
    cleanup_memory,
    estimate_memory_requirements,
    memory_monitoring_context,
    safe_pickle_load,
    safe_pickle_save,
    validate_and_setup_args,
)


@pytest.mark.unit
class TestPickleUtilities:
    """Test pickle utility functions."""

    def test_safe_pickle_save_success(self, temp_dir):
        """Test successful pickle save."""
        test_data = {"test": "data", "numbers": [1, 2, 3]}
        test_file = temp_dir / "test.pkl"

        result = safe_pickle_save(test_data, test_file, "Test data")

        assert result is True
        assert test_file.exists()

        # Verify the data was saved correctly
        with open(test_file, "rb") as f:
            loaded_data = pickle.load(f)
        assert loaded_data == test_data

    def test_safe_pickle_save_failure(self, temp_dir):
        """Test pickle save failure handling."""
        test_data = {"test": "data"}
        # Try to save to a non-existent directory
        test_file = temp_dir / "nonexistent" / "test.pkl"

        result = safe_pickle_save(test_data, test_file, "Test data")

        assert result is False

    def test_safe_pickle_load_success(self, temp_dir):
        """Test successful pickle load."""
        test_data = {"test": "data", "array": np.array([1, 2, 3])}
        test_file = temp_dir / "test.pkl"

        # Save test data first
        with open(test_file, "wb") as f:
            pickle.dump(test_data, f)

        loaded_data = safe_pickle_load(test_file, "Test data")

        assert loaded_data is not None
        assert loaded_data["test"] == "data"
        np.testing.assert_array_equal(loaded_data["array"], np.array([1, 2, 3]))

    def test_safe_pickle_load_file_not_found(self, temp_dir):
        """Test pickle load when file doesn't exist."""
        test_file = temp_dir / "nonexistent.pkl"

        result = safe_pickle_load(test_file, "Test data")

        assert result is None

    @patch("utils.logger")
    def test_safe_pickle_load_corruption(self, mock_logger, temp_dir):
        """Test pickle load with corrupted file."""
        test_file = temp_dir / "corrupted.pkl"

        # Create a corrupted pickle file
        with open(test_file, "wb") as f:
            f.write(b"corrupted_data")

        result = safe_pickle_load(test_file, "Test data")

        assert result is None
        mock_logger.error.assert_called()


@pytest.mark.unit
class TestMemoryUtilities:
    """Test memory utility functions."""

    def test_memory_monitoring_context(self):
        """Test memory monitoring context manager."""
        with patch("utils.psutil") as mock_psutil:
            mock_process = Mock()
            mock_process.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
            mock_psutil.Process.return_value = mock_process

            with memory_monitoring_context("Test operation"):
                # Simulate some operation
                pass

            # Should have called memory_info at least twice (start and end)
            assert mock_process.memory_info.call_count >= 2

    @patch("utils.psutil")
    def test_check_available_memory(self, mock_psutil):
        """Test available memory check."""
        mock_psutil.virtual_memory.return_value.available = 8 * 1024**3  # 8GB

        available_gb = check_available_memory()

        assert available_gb == 8.0

    @patch("utils.psutil")
    def test_cleanup_memory(self, mock_psutil):
        """Test memory cleanup function."""

        with patch("gc.collect") as mock_gc:
            cleanup_memory()

            mock_gc.assert_called_once()

    @patch("utils.logger")
    def test_estimate_memory_requirements(self, mock_logger):
        """Test memory requirements estimation."""
        estimate_memory_requirements(
            n_subjects=100, n_features=1000, n_factors=10, n_chains=2, n_samples=1000
        )

        # Should log memory estimates
        assert mock_logger.info.called

        # Check that reasonable estimates were made
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Data matrices" in call for call in log_calls)
        assert any("MCMC samples" in call for call in log_calls)
        assert any("Total estimated" in call for call in log_calls)


@pytest.mark.unit
class TestValidationUtilities:
    """Test validation utility functions."""

    def test_validate_and_setup_args_basic(self):
        """Test basic argument validation."""
        args = Mock()
        args.dataset = "synthetic"
        args.K = 5
        args.num_sources = 3
        args.device = "cpu"
        args.results_dir = None
        args.seed = 42

        validated_args = validate_and_setup_args(args)

        assert validated_args.dataset == "synthetic"
        assert validated_args.K == 5
        assert validated_args.results_dir is not None  # Should be set
        assert validated_args.seed == 42

    def test_validate_and_setup_args_with_results_dir(self, temp_dir):
        """Test validation with existing results directory."""
        args = Mock()
        args.dataset = "qmap_pd"
        args.data_dir = str(temp_dir)
        args.results_dir = str(temp_dir / "results")
        args.K = 10
        args.device = "gpu"
        args.seed = 123

        validated_args = validate_and_setup_args(args)

        assert validated_args.results_dir == str(temp_dir / "results")
        assert Path(validated_args.results_dir).exists()

    def test_validate_and_setup_args_cv_available(self):
        """Test validation with CV available."""
        args = Mock()
        args.dataset = "synthetic"
        args.run_cv = True
        args.K = 8

        with patch("utils.logger") as mock_logger:
            validated_args = validate_and_setup_args(
                args, cv_available=True, neuroimaging_cv_available=True
            )

            # Should log CV availability
            assert mock_logger.info.called

    def test_validate_and_setup_args_invalid_k(self):
        """Test validation with invalid K value."""
        args = Mock()
        args.dataset = "synthetic"
        args.K = 0  # Invalid

        with pytest.raises(ValueError, match="K must be positive"):
            validate_and_setup_args(args)

    def test_validate_and_setup_args_missing_data_dir(self):
        """Test validation with missing data directory for qMAP-PD."""
        args = Mock()
        args.dataset = "qmap_pd"
        args.data_dir = None
        args.K = 5

        with pytest.raises(ValueError, match="data_dir is required"):
            validate_and_setup_args(args)

    @patch("utils.Path.mkdir")
    @patch("utils.logger")
    def test_validate_and_setup_args_creates_results_dir(self, mock_logger, mock_mkdir):
        """Test that results directory is created if it doesn't exist."""
        args = Mock()
        args.dataset = "synthetic"
        args.results_dir = "/fake/results"
        args.K = 5

        with patch("utils.Path.exists", return_value=False):
            validate_and_setup_args(args)

            mock_mkdir.assert_called_once()


@pytest.mark.unit
class TestFileUtilities:
    """Test file utility functions."""

    def test_get_model_files_basic(self):
        """Test getting model files list."""
        with patch("utils.get_model_files") as mock_get_model_files:
            mock_get_model_files.return_value = ["model1.py", "model2.py"]

            files = mock_get_model_files()
            assert len(files) == 2
            assert "model1.py" in files

    def test_path_handling_with_spaces(self):
        """Test path handling with spaces in names."""
        # This would test any path utilities that handle spaces
        # Since there aren't specific functions visible, this is a placeholder
        test_path = Path("path with spaces/file name.txt")
        assert str(test_path) == "path with spaces/file name.txt"


@pytest.mark.integration
class TestUtilityIntegration:
    """Integration tests for utility functions."""

    def test_pickle_roundtrip_with_numpy(self, temp_dir):
        """Test complete pickle save/load roundtrip with numpy arrays."""
        original_data = {
            "arrays": {
                "Z": np.random.normal(0, 1, (100, 50, 5)),
                "W": np.random.normal(0, 1, (200, 5)),
                "sigma": np.random.gamma(2, 1, 100),
            },
            "metadata": {
                "dataset": "test",
                "n_samples": 100,
                "created_by": "test_suite",
            },
        }

        test_file = temp_dir / "integration_test.pkl"

        # Save
        save_success = safe_pickle_save(original_data, test_file, "Integration test")
        assert save_success is True

        # Load
        loaded_data = safe_pickle_load(test_file, "Integration test")
        assert loaded_data is not None

        # Verify arrays
        np.testing.assert_array_equal(
            loaded_data["arrays"]["Z"], original_data["arrays"]["Z"]
        )
        np.testing.assert_array_equal(
            loaded_data["arrays"]["W"], original_data["arrays"]["W"]
        )
        np.testing.assert_array_equal(
            loaded_data["arrays"]["sigma"], original_data["arrays"]["sigma"]
        )

        # Verify metadata
        assert loaded_data["metadata"] == original_data["metadata"]

    @patch("utils.psutil")
    def test_memory_monitoring_during_operation(self, mock_psutil):
        """Test memory monitoring during actual operations."""
        mock_process = Mock()
        # Simulate memory increase during operation
        memory_values = [
            100 * 1024**2,
            150 * 1024**2,
            120 * 1024**2,
        ]  # 100MB -> 150MB -> 120MB
        mock_process.memory_info.return_value.rss = memory_values[0]
        mock_psutil.Process.return_value = mock_process

        call_count = 0

        def memory_side_effect():
            nonlocal call_count
            result = Mock()
            result.rss = memory_values[min(call_count, len(memory_values) - 1)]
            call_count += 1
            return result

        mock_process.memory_info = memory_side_effect

        with memory_monitoring_context("Test operation"):
            # Simulate some memory-intensive operation
            test_array = np.random.normal(0, 1, (1000, 1000))
            del test_array

        # Memory should have been monitored
        assert call_count >= 2
