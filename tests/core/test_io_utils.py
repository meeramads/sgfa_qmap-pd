"""Tests for core.io_utils module."""

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.io_utils import (
    FileOperationManager,
    ensure_directory,
    load_csv,
    load_json,
    load_numpy,
    load_pickle,
    safe_filename,
    save_csv,
    save_json,
    save_numpy,
    save_pickle,
    save_plot,
)


class TestSaveLoadJSON:
    """Test JSON save/load functions."""

    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading JSON data."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        filepath = tmp_path / "test.json"

        save_json(data, filepath)
        assert filepath.exists()

        loaded = load_json(filepath)
        assert loaded == data

    def test_save_json_creates_directory(self, tmp_path):
        """Test that save_json creates parent directories."""
        filepath = tmp_path / "subdir" / "test.json"
        data = {"key": "value"}

        save_json(data, filepath)

        assert filepath.exists()
        assert filepath.parent.exists()

    def test_save_json_with_indent(self, tmp_path):
        """Test JSON saving with custom indentation."""
        data = {"key": "value"}
        filepath = tmp_path / "test.json"

        save_json(data, filepath, indent=4)

        with open(filepath, "r") as f:
            content = f.read()

        # Check that content is indented (contains newlines and spaces)
        assert "\n" in content
        assert "    " in content

    def test_save_json_handles_non_serializable_with_default(self, tmp_path):
        """Test that save_json handles non-serializable objects with default=str."""
        data = {"date": pd.Timestamp("2024-01-01")}
        filepath = tmp_path / "test.json"

        # Should not raise - converts to string
        save_json(data, filepath)
        loaded = load_json(filepath)

        assert "date" in loaded

    def test_load_json_nonexistent_raises(self, tmp_path):
        """Test that loading nonexistent JSON raises error."""
        filepath = tmp_path / "nonexistent.json"

        with pytest.raises(Exception):
            load_json(filepath)

    def test_save_json_accepts_string_path(self, tmp_path):
        """Test that save_json accepts string paths."""
        data = {"key": "value"}
        filepath = str(tmp_path / "test.json")

        save_json(data, filepath)
        assert Path(filepath).exists()


class TestSaveLoadCSV:
    """Test CSV save/load functions."""

    def test_save_and_load_csv(self, tmp_path):
        """Test saving and loading CSV data."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        filepath = tmp_path / "test.csv"

        save_csv(df, filepath)
        assert filepath.exists()

        loaded = load_csv(filepath)
        pd.testing.assert_frame_equal(loaded, df)

    def test_save_csv_without_index(self, tmp_path):
        """Test saving CSV without index (default behavior)."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        filepath = tmp_path / "test.csv"

        save_csv(df, filepath)

        with open(filepath, "r") as f:
            content = f.read()

        # Index column should not be present
        assert not content.startswith(",col")

    def test_save_csv_with_index(self, tmp_path):
        """Test saving CSV with index."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        filepath = tmp_path / "test.csv"

        save_csv(df, filepath, index=True)

        with open(filepath, "r") as f:
            first_line = f.readline()

        # Index column should be present
        assert first_line.startswith(",col") or "Unnamed" in first_line

    def test_save_csv_creates_directory(self, tmp_path):
        """Test that save_csv creates parent directories."""
        filepath = tmp_path / "subdir" / "test.csv"
        df = pd.DataFrame({"col": [1, 2, 3]})

        save_csv(df, filepath)

        assert filepath.exists()

    def test_load_csv_with_kwargs(self, tmp_path):
        """Test loading CSV with additional kwargs."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        filepath = tmp_path / "test.csv"

        save_csv(df, filepath)
        loaded = load_csv(filepath, dtype={"col": float})

        assert loaded["col"].dtype == float


class TestSaveLoadNumpy:
    """Test numpy array save/load functions."""

    def test_save_and_load_numpy_compressed(self, tmp_path):
        """Test saving and loading compressed numpy array."""
        data = np.random.randn(10, 5)
        filepath = tmp_path / "test.npz"

        save_numpy(data, filepath, compressed=True)
        assert filepath.exists()

        loaded = load_numpy(filepath)
        np.testing.assert_array_equal(loaded, data)

    def test_save_and_load_numpy_uncompressed(self, tmp_path):
        """Test saving and loading uncompressed numpy array."""
        data = np.random.randn(10, 5)
        filepath = tmp_path / "test.npy"

        save_numpy(data, filepath, compressed=False)
        assert filepath.exists()

        loaded = load_numpy(filepath)
        np.testing.assert_array_equal(loaded, data)

    def test_save_numpy_adds_correct_extension(self, tmp_path):
        """Test that save_numpy adds correct extension."""
        data = np.random.randn(5, 3)

        # Compressed should use .npz
        filepath1 = tmp_path / "test1"
        save_numpy(data, filepath1, compressed=True)
        assert (tmp_path / "test1.npz").exists()

        # Uncompressed should use .npy
        filepath2 = tmp_path / "test2"
        save_numpy(data, filepath2, compressed=False)
        assert (tmp_path / "test2.npy").exists()

    def test_save_numpy_large_array_warning(self, tmp_path, caplog):
        """Test warning for large arrays."""
        # Create large array (> default 500MB would be too big, use small max_size for test)
        data = np.random.randn(1000, 1000)  # ~8MB
        filepath = tmp_path / "test.npy"

        save_numpy(data, filepath, compressed=False, max_size_mb=1)

        # Check for warning
        assert "Large array detected" in caplog.text

    def test_load_numpy_nonexistent_raises(self, tmp_path):
        """Test that loading nonexistent numpy file raises error."""
        filepath = tmp_path / "nonexistent.npy"

        with pytest.raises(Exception):
            load_numpy(filepath)


class TestSaveLoadPickle:
    """Test pickle save/load functions."""

    def test_save_and_load_pickle(self, tmp_path):
        """Test saving and loading pickle data."""
        data = {"key": "value", "list": [1, 2, 3], "array": np.array([1, 2, 3])}
        filepath = tmp_path / "test.pkl"

        save_pickle(data, filepath)
        assert filepath.exists()

        loaded = load_pickle(filepath)

        assert loaded["key"] == data["key"]
        assert loaded["list"] == data["list"]
        np.testing.assert_array_equal(loaded["array"], data["array"])

    def test_save_pickle_creates_directory(self, tmp_path):
        """Test that save_pickle creates parent directories."""
        filepath = tmp_path / "subdir" / "test.pkl"
        data = {"key": "value"}

        save_pickle(data, filepath)

        assert filepath.exists()

    def test_pickle_preserves_complex_objects(self, tmp_path):
        """Test that pickle preserves complex objects."""
        class CustomClass:
            def __init__(self, value):
                self.value = value

        data = CustomClass(42)
        filepath = tmp_path / "test.pkl"

        save_pickle(data, filepath)
        loaded = load_pickle(filepath)

        assert isinstance(loaded, CustomClass)
        assert loaded.value == 42


class TestSavePlot:
    """Test save_plot function."""

    @patch("core.io_utils.plt")
    def test_save_plot_basic(self, mock_plt, tmp_path):
        """Test basic plot saving."""
        filepath = tmp_path / "plot.png"

        save_plot(filepath)

        mock_plt.savefig.assert_called_once()
        assert mock_plt.savefig.call_args[0][0] == filepath

    @patch("core.io_utils.plt")
    def test_save_plot_custom_dpi(self, mock_plt, tmp_path):
        """Test saving plot with custom DPI."""
        filepath = tmp_path / "plot.png"

        save_plot(filepath, dpi=150)

        assert mock_plt.savefig.call_args[1]["dpi"] == 150

    @patch("core.io_utils.plt")
    def test_save_plot_closes_figure(self, mock_plt, tmp_path):
        """Test that plot is closed after saving."""
        filepath = tmp_path / "plot.png"

        save_plot(filepath, close_after=True)

        mock_plt.close.assert_called_once()

    @patch("core.io_utils.plt")
    def test_save_plot_no_close(self, mock_plt, tmp_path):
        """Test that plot is not closed when close_after=False."""
        filepath = tmp_path / "plot.png"

        save_plot(filepath, close_after=False)

        mock_plt.close.assert_not_called()

    @patch("core.io_utils.plt")
    def test_save_plot_creates_directory(self, mock_plt, tmp_path):
        """Test that save_plot creates parent directories."""
        filepath = tmp_path / "subdir" / "plot.png"

        save_plot(filepath)

        assert filepath.parent.exists()

    @patch("core.io_utils.plt")
    def test_save_plot_closes_on_error(self, mock_plt, tmp_path):
        """Test that plot is closed even when save fails."""
        mock_plt.savefig.side_effect = Exception("Save failed")
        filepath = tmp_path / "plot.png"

        with pytest.raises(Exception):
            save_plot(filepath, close_after=True)

        # Should still close on error
        mock_plt.close.assert_called_once()


class TestEnsureDirectory:
    """Test ensure_directory function."""

    def test_ensure_directory_creates_new(self, tmp_path):
        """Test creating new directory."""
        dirpath = tmp_path / "newdir"

        result = ensure_directory(dirpath)

        assert dirpath.exists()
        assert dirpath.is_dir()
        assert result == dirpath

    def test_ensure_directory_existing(self, tmp_path):
        """Test with existing directory."""
        dirpath = tmp_path / "existingdir"
        dirpath.mkdir()

        result = ensure_directory(dirpath)

        assert dirpath.exists()
        assert result == dirpath

    def test_ensure_directory_creates_parents(self, tmp_path):
        """Test creating nested directories."""
        dirpath = tmp_path / "parent" / "child" / "grandchild"

        result = ensure_directory(dirpath)

        assert dirpath.exists()
        assert dirpath.is_dir()

    def test_ensure_directory_accepts_string(self, tmp_path):
        """Test that ensure_directory accepts string paths."""
        dirpath = str(tmp_path / "newdir")

        result = ensure_directory(dirpath)

        assert Path(dirpath).exists()
        assert isinstance(result, Path)


class TestSafeFilename:
    """Test safe_filename function."""

    def test_safe_filename_no_changes(self):
        """Test filename with no problematic characters."""
        filename = "normal_filename.txt"

        result = safe_filename(filename)

        assert result == filename

    def test_safe_filename_replaces_special_chars(self):
        """Test replacing special characters."""
        filename = "file<name>with:special*chars?.txt"

        result = safe_filename(filename)

        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "*" not in result
        assert "?" not in result

    def test_safe_filename_custom_replacement(self):
        """Test with custom replacement character."""
        filename = "file:name.txt"

        result = safe_filename(filename, replacement="-")

        assert result == "file-name.txt"

    def test_safe_filename_multiple_consecutive(self):
        """Test that multiple consecutive special chars become single replacement."""
        filename = "file:::name.txt"

        result = safe_filename(filename, replacement="_")

        assert result == "file_name.txt"

    def test_safe_filename_strips_replacement(self):
        """Test that leading/trailing replacements are stripped."""
        filename = ":filename:"

        result = safe_filename(filename, replacement="_")

        assert result == "filename"

    def test_safe_filename_path_separators(self):
        """Test replacing path separators."""
        filename = "path/to\\file.txt"

        result = safe_filename(filename)

        assert "/" not in result
        assert "\\" not in result


class TestFileOperationManager:
    """Test FileOperationManager context manager."""

    def test_file_operation_manager_initialization(self, tmp_path):
        """Test FileOperationManager initialization."""
        with FileOperationManager(tmp_path / "output") as manager:
            assert manager.base_dir == tmp_path / "output"
            assert manager.base_dir.exists()
            assert manager.files_created == []

    def test_file_operation_manager_save_json(self, tmp_path):
        """Test saving JSON through manager."""
        with FileOperationManager(tmp_path / "output") as manager:
            data = {"key": "value"}
            filepath = manager.save_json(data, "test.json")

            assert filepath.exists()
            assert filepath in manager.files_created
            assert filepath.parent == manager.base_dir

    def test_file_operation_manager_save_csv(self, tmp_path):
        """Test saving CSV through manager."""
        with FileOperationManager(tmp_path / "output") as manager:
            df = pd.DataFrame({"col": [1, 2, 3]})
            filepath = manager.save_csv(df, "test.csv")

            assert filepath.exists()
            assert filepath in manager.files_created

    def test_file_operation_manager_save_numpy(self, tmp_path):
        """Test saving numpy array through manager."""
        with FileOperationManager(tmp_path / "output") as manager:
            data = np.random.randn(10, 5)
            filepath = manager.save_numpy(data, "test.npz")

            assert filepath.exists()
            assert filepath in manager.files_created

    @patch("core.io_utils.save_plot")
    def test_file_operation_manager_save_plot(self, mock_save_plot, tmp_path):
        """Test saving plot through manager."""
        with FileOperationManager(tmp_path / "output") as manager:
            filepath = manager.save_plot("plot.png")

            mock_save_plot.assert_called_once()
            assert filepath in manager.files_created

    def test_file_operation_manager_tracks_files(self, tmp_path):
        """Test that manager tracks all created files."""
        with FileOperationManager(tmp_path / "output") as manager:
            manager.save_json({"a": 1}, "file1.json")
            manager.save_json({"b": 2}, "file2.json")
            df = pd.DataFrame({"col": [1, 2]})
            manager.save_csv(df, "file3.csv")

            assert len(manager.files_created) == 3

    def test_file_operation_manager_handles_errors(self, tmp_path, caplog):
        """Test that manager handles errors gracefully."""
        try:
            with FileOperationManager(tmp_path / "output") as manager:
                manager.save_json({"key": "value"}, "test.json")
                raise Exception("Test error")
        except Exception:
            pass

        # Check that warning was logged
        assert "exited with error" in caplog.text


class TestDataManagerAlias:
    """Test DataManager backward compatibility alias."""

    def test_data_manager_is_alias(self):
        """Test that DataManager is an alias for FileOperationManager."""
        from core.io_utils import DataManager

        assert DataManager is FileOperationManager

    def test_data_manager_works(self, tmp_path):
        """Test that DataManager alias works."""
        from core.io_utils import DataManager

        with DataManager(tmp_path / "output") as manager:
            assert isinstance(manager, FileOperationManager)


class TestIOIntegration:
    """Integration tests for IO utilities."""

    def test_full_workflow(self, tmp_path):
        """Test complete save/load workflow for all formats."""
        output_dir = tmp_path / "output"

        with FileOperationManager(output_dir) as manager:
            # Save various formats
            manager.save_json({"key": "value"}, "config.json")
            manager.save_csv(pd.DataFrame({"col": [1, 2, 3]}), "data.csv")
            manager.save_numpy(np.array([1, 2, 3]), "array.npz")

            # Check all files created
            assert len(manager.files_created) == 3

        # Load everything back
        loaded_json = load_json(output_dir / "config.json")
        loaded_csv = load_csv(output_dir / "data.csv")
        loaded_numpy = load_numpy(output_dir / "array.npz")

        assert loaded_json == {"key": "value"}
        assert len(loaded_csv) == 3
        np.testing.assert_array_equal(loaded_numpy, np.array([1, 2, 3]))

    def test_nested_directory_creation(self, tmp_path):
        """Test creating deeply nested directory structures."""
        filepath = tmp_path / "a" / "b" / "c" / "d" / "file.json"

        save_json({"nested": True}, filepath)

        assert filepath.exists()
        assert load_json(filepath) == {"nested": True}

    def test_path_type_consistency(self, tmp_path):
        """Test that Path and str paths work consistently."""
        data = {"test": "data"}

        # Save with Path
        path_obj = tmp_path / "path_test.json"
        save_json(data, path_obj)

        # Save with str
        str_path = str(tmp_path / "str_test.json")
        save_json(data, str_path)

        # Both should work
        assert load_json(path_obj) == data
        assert load_json(str_path) == data
