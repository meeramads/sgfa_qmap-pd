"""IO utilities for consistent file operations across experiments."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file with consistent formatting.

    Parameters
    ----------
    data : Any
        Data to save (must be JSON serializable)
    filepath : Union[str, Path]
        Output file path
    indent : int, optional
        JSON indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent, default=str)
        logger.debug(f"Saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        raise


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from JSON file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Input file path

    Returns
    -------
    Any
        Loaded data
    """
    filepath = Path(filepath)
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        raise


def save_csv(
    df: pd.DataFrame, filepath: Union[str, Path], index: bool = False, **kwargs
) -> None:
    """
    Save DataFrame to CSV with consistent formatting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : Union[str, Path]
        Output file path
    index : bool, optional
        Whether to save index
    **kwargs
        Additional arguments for to_csv
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(filepath, index=index, **kwargs)
        logger.debug(f"Saved CSV to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save CSV to {filepath}: {e}")
        raise


def load_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from CSV.

    Parameters
    ----------
    filepath : Union[str, Path]
        Input file path
    **kwargs
        Additional arguments for read_csv

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    filepath = Path(filepath)
    try:
        df = pd.read_csv(filepath, **kwargs)
        logger.debug(f"Loaded CSV from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV from {filepath}: {e}")
        raise


def save_numpy(
    data: np.ndarray, filepath: Union[str, Path], compressed: bool = True, max_size_mb: int = 500
) -> None:
    """
    Save numpy array to file.

    Parameters
    ----------
    data : np.ndarray
        Array to save
    filepath : Union[str, Path]
        Output file path
    compressed : bool, optional
        Whether to use compressed format (.npz vs .npy)
    max_size_mb : int, optional
        Maximum file size in MB before warning (default: 500)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Check array size before saving
    array_size_mb = data.nbytes / (1024 * 1024)
    if array_size_mb > max_size_mb:
        logger.warning(f"Large array detected: {array_size_mb:.1f}MB > {max_size_mb}MB limit for {filepath}")
        logger.warning("Consider using chunked processing or reducing data size")

    try:
        if compressed:
            # Use .npz extension for compressed
            if not str(filepath).endswith(".npz"):
                filepath = filepath.with_suffix(".npz")
            np.savez_compressed(filepath, data=data)
        else:
            # Use .npy extension for uncompressed
            if not str(filepath).endswith(".npy"):
                filepath = filepath.with_suffix(".npy")
            np.save(filepath, data)
        logger.debug(f"Saved numpy array to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save numpy array to {filepath}: {e}")
        raise


def load_numpy(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load numpy array from file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Input file path

    Returns
    -------
    np.ndarray
        Loaded array
    """
    filepath = Path(filepath)
    try:
        if str(filepath).endswith(".npz"):
            # Load from compressed format
            with np.load(filepath) as data:
                # Assume single array stored as 'data'
                return data["data"]
        else:
            # Load from uncompressed format
            data = np.load(filepath)
        logger.debug(f"Loaded numpy array from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load numpy array from {filepath}: {e}")
        raise


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """
    Save data to pickle file.

    Parameters
    ----------
    data : Any
        Data to save
    filepath : Union[str, Path]
        Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"Saved pickle to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save pickle to {filepath}: {e}")
        raise


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load data from pickle file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Input file path

    Returns
    -------
    Any
        Loaded data
    """
    filepath = Path(filepath)
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        logger.debug(f"Loaded pickle from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load pickle from {filepath}: {e}")
        raise


def save_plot(
    filepath: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = "tight",
    close_after: bool = True,
    **kwargs,
) -> None:
    """
    Save matplotlib plot with consistent formatting.

    Parameters
    ----------
    filepath : Union[str, Path]
        Output file path
    dpi : int, optional
        Resolution in dots per inch
    bbox_inches : str, optional
        Bounding box handling
    close_after : bool, optional
        Whether to close the figure after saving
    **kwargs
        Additional arguments for savefig
    """
    import matplotlib.pyplot as plt

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        logger.debug(f"Saved plot to {filepath}")

        if close_after:
            plt.close()

    except Exception as e:
        logger.error(f"Failed to save plot to {filepath}: {e}")
        if close_after:
            plt.close()  # Clean up even on error
        raise


def ensure_directory(dirpath: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating if necessary.

    Parameters
    ----------
    dirpath : Union[str, Path]
        Directory path to ensure

    Returns
    -------
    Path
        Resolved directory path
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {dirpath}")
    return dirpath


def safe_filename(filename: str, replacement: str = "_") -> str:
    """
    Create safe filename by replacing problematic characters.

    Parameters
    ----------
    filename : str
        Original filename
    replacement : str, optional
        Character to replace problematic characters with

    Returns
    -------
    str
        Safe filename
    """
    import re

    # Replace problematic characters with replacement
    safe_name = re.sub(r'[<>:"/\\|?*]', replacement, filename)
    # Remove multiple consecutive replacements
    safe_name = re.sub(f"{re.escape(replacement)}+", replacement, safe_name)
    return safe_name.strip(replacement)


class DataManager:
    """
    Context manager for handling multiple file operations safely.

    Automatically creates directories and handles errors consistently.
    """

    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        self.files_created = []

    def __enter__(self):
        ensure_directory(self.base_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.warning(f"DataManager exited with error: {exc_val}")
        logger.debug(f"DataManager handled {len(self.files_created)} files")

    def save_json(self, data: Any, filename: str, **kwargs) -> Path:
        """Save JSON file within managed directory."""
        filepath = self.base_dir / filename
        save_json(data, filepath, **kwargs)
        self.files_created.append(filepath)
        return filepath

    def save_csv(self, df: pd.DataFrame, filename: str, **kwargs) -> Path:
        """Save CSV file within managed directory."""
        filepath = self.base_dir / filename
        save_csv(df, filepath, **kwargs)
        self.files_created.append(filepath)
        return filepath

    def save_numpy(self, data: np.ndarray, filename: str, **kwargs) -> Path:
        """Save numpy file within managed directory."""
        filepath = self.base_dir / filename
        save_numpy(data, filepath, **kwargs)
        self.files_created.append(filepath)
        return filepath

    def save_plot(self, filename: str, **kwargs) -> Path:
        """Save plot within managed directory."""
        filepath = self.base_dir / filename
        save_plot(filepath, **kwargs)
        self.files_created.append(filepath)
        return filepath
