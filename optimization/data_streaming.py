"""Data streaming and chunking utilities for memory-efficient processing."""

import logging
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union

import h5py
import numpy as np
import pandas as pd

from .memory_optimizer import MemoryOptimizer, adaptive_batch_size

logger = logging.getLogger(__name__)


class DataStreamer:
    """Memory-efficient data streaming for large datasets."""

    def __init__(
        self,
        chunk_size: int = None,
        memory_limit_gb: float = 4.0,
        preload_next: bool = True,
        cache_chunks: bool = False,
    ):
        """
        Initialize data streamer.

        Parameters
        ----------
        chunk_size : int, optional
            Number of samples per chunk. If None, calculated adaptively.
        memory_limit_gb : float
            Memory limit for chunk sizing.
        preload_next : bool
            Whether to preload next chunk while processing current.
        cache_chunks : bool
            Whether to cache processed chunks in memory.
        """
        self.chunk_size = chunk_size
        self.memory_limit_gb = memory_limit_gb
        self.preload_next = preload_next
        self.cache_chunks = cache_chunks
        self.chunk_cache = {} if cache_chunks else None
        self._memory_optimizer = MemoryOptimizer()

        logger.info(
            f"DataStreamer initialized: chunk_size={chunk_size}, "
            f"memory_limit={memory_limit_gb}GB"
        )

    def stream_array_chunks(
        self, arrays: List[np.ndarray], axis: int = 0
    ) -> Iterator[List[np.ndarray]]:
        """
        Stream array data in memory-efficient chunks.

        Parameters
        ----------
        arrays : List[np.ndarray]
            List of arrays to stream (must have same size along axis).
        axis : int
            Axis along which to chunk.

        Yields
        ------
        List[np.ndarray] : Chunk of arrays.
        """
        if not arrays:
            return

        # Validate arrays have same size along chunking axis
        chunk_dim = arrays[0].shape[axis]
        for i, arr in enumerate(arrays[1:], 1):
            if arr.shape[axis] != chunk_dim:
                raise ValueError(f"Array {i} has different size along axis {axis}")

        # Calculate optimal chunk size if not provided
        if self.chunk_size is None:
            element_size = sum(
                arr.dtype.itemsize * np.prod(arr.shape) // arr.shape[axis]
                for arr in arrays
            )
            chunk_size = adaptive_batch_size(
                chunk_dim,
                self.memory_limit_gb,
                element_size,
                min_batch=1,
                max_batch=chunk_dim,
            )
        else:
            chunk_size = min(self.chunk_size, chunk_dim)

        logger.debug(f"Streaming {len(arrays)} arrays with chunk_size={chunk_size}")

        # Stream chunks
        for start_idx in range(0, chunk_dim, chunk_size):
            end_idx = min(start_idx + chunk_size, chunk_dim)

            # Create chunk key for caching
            chunk_key = f"chunk_{start_idx}_{end_idx}" if self.cache_chunks else None

            # Check cache first
            if self.cache_chunks and chunk_key in self.chunk_cache:
                logger.debug(f"Using cached chunk {chunk_key}")
                yield self.chunk_cache[chunk_key]
                continue

            # Extract chunk from each array
            chunks = []
            for arr in arrays:
                if axis == 0:
                    chunk = arr[start_idx:end_idx]
                elif axis == 1:
                    chunk = arr[:, start_idx:end_idx]
                else:
                    # General case using advanced indexing
                    slice_obj = tuple(
                        slice(start_idx, end_idx) if i == axis else slice(None)
                        for i in range(arr.ndim)
                    )
                    chunk = arr[slice_obj]

                chunks.append(chunk)

            # Cache if enabled
            if self.cache_chunks:
                self.chunk_cache[chunk_key] = chunks

            yield chunks

            # Memory cleanup after each chunk
            if start_idx % (chunk_size * 10) == 0:  # Every 10 chunks
                self._memory_optimizer.aggressive_cleanup()

    def stream_hdf5_dataset(
        self, filepath: Union[str, Path], dataset_names: List[str], chunk_axis: int = 0
    ) -> Iterator[Dict[str, np.ndarray]]:
        """
        Stream HDF5 datasets in chunks.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to HDF5 file.
        dataset_names : List[str]
            Names of datasets to stream.
        chunk_axis : int
            Axis along which to chunk datasets.

        Yields
        ------
        Dict[str, np.ndarray] : Chunk of named datasets.
        """
        with h5py.File(filepath, "r") as f:
            # Get dataset shapes
            datasets = {name: f[name] for name in dataset_names}

            # Validate shapes
            chunk_dim = datasets[dataset_names[0]].shape[chunk_axis]
            for name, dataset in datasets.items():
                if dataset.shape[chunk_axis] != chunk_dim:
                    raise ValueError(
                        f"Dataset {name} has different size along axis {chunk_axis}"
                    )

            # Calculate chunk size
            if self.chunk_size is None:
                element_size = sum(
                    dataset.dtype.itemsize
                    * np.prod(dataset.shape)
                    // dataset.shape[chunk_axis]
                    for dataset in datasets.values()
                )
                chunk_size = adaptive_batch_size(
                    chunk_dim, self.memory_limit_gb, element_size, min_batch=1
                )
            else:
                chunk_size = min(self.chunk_size, chunk_dim)

            logger.info(f"Streaming HDF5 datasets with chunk_size={chunk_size}")

            # Stream chunks
            for start_idx in range(0, chunk_dim, chunk_size):
                end_idx = min(start_idx + chunk_size, chunk_dim)

                chunk_data = {}
                for name, dataset in datasets.items():
                    if chunk_axis == 0:
                        chunk_data[name] = dataset[start_idx:end_idx]
                    else:
                        # General slicing
                        slice_obj = tuple(
                            (
                                slice(start_idx, end_idx)
                                if i == chunk_axis
                                else slice(None)
                            )
                            for i in range(dataset.ndim)
                        )
                        chunk_data[name] = dataset[slice_obj]

                yield chunk_data

    def clear_cache(self):
        """Clear chunk cache."""
        if self.chunk_cache:
            self.chunk_cache.clear()
            logger.debug("Cleared chunk cache")


class ChunkedDataLoader:
    """Enhanced data loader with chunking and memory optimization."""

    def __init__(
        self,
        memory_limit_gb: float = 4.0,
        enable_compression: bool = True,
        compression_level: int = 6,
        parallel_loading: bool = False,
    ):
        """
        Initialize chunked data loader.

        Parameters
        ----------
        memory_limit_gb : float
            Memory limit for chunk operations.
        enable_compression : bool
            Enable data compression for memory savings.
        compression_level : int
            Compression level (1-9, higher = better compression).
        parallel_loading : bool
            Enable parallel chunk loading.
        """
        self.memory_limit_gb = memory_limit_gb
        self.enable_compression = enable_compression
        self.compression_level = compression_level
        self.parallel_loading = parallel_loading
        self._streamer = DataStreamer(memory_limit_gb=memory_limit_gb)

    def load_multiview_data_chunked(
        self, data: Dict[str, Any], chunk_subjects: int = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Load multi-view data in memory-efficient chunks by subjects.

        Parameters
        ----------
        data : Dict[str, Any]
            Multi-view data dictionary with X_list, clinical, etc.
        chunk_subjects : int, optional
            Number of subjects per chunk.

        Yields
        ------
        Dict[str, Any] : Chunked data dictionary.
        """
        X_list = data["X_list"]
        n_subjects = X_list[0].shape[0]

        # Calculate optimal chunk size if not provided
        if chunk_subjects is None:
            total_features = sum(X.shape[1] for X in X_list)
            element_size = total_features * X_list[0].dtype.itemsize
            chunk_subjects = adaptive_batch_size(
                n_subjects,
                self.memory_limit_gb,
                element_size,
                min_batch=10,  # Minimum for meaningful statistics
                max_batch=n_subjects,
            )

        logger.info(f"Loading data in chunks of {chunk_subjects} subjects")

        # Stream chunks
        for start_idx in range(0, n_subjects, chunk_subjects):
            end_idx = min(start_idx + chunk_subjects, n_subjects)
            subject_slice = slice(start_idx, end_idx)

            # Create chunked data dictionary
            chunked_data = {
                "X_list": [X[subject_slice] for X in X_list],
                "view_names": data["view_names"],
                "feature_names": data["feature_names"],
                "subject_ids": data["subject_ids"][start_idx:end_idx],
                "chunk_info": {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "chunk_size": end_idx - start_idx,
                    "total_subjects": n_subjects,
                    "chunk_id": start_idx // chunk_subjects,
                },
            }

            # Add clinical data if present
            if "clinical" in data and data["clinical"] is not None:
                chunked_data["clinical"] = data["clinical"].iloc[subject_slice]

            # Add scalers (same for all chunks)
            if "scalers" in data:
                chunked_data["scalers"] = data["scalers"]

            # Add metadata
            if "meta" in data:
                chunked_data["meta"] = data["meta"].copy()
                chunked_data["meta"]["N"] = end_idx - start_idx
                chunked_data["meta"]["chunk_info"] = chunked_data["chunk_info"]

            # Compress if enabled
            if self.enable_compression:
                chunked_data = self._compress_chunk(chunked_data)

            yield chunked_data

    def _compress_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply compression to chunk data."""
        # For now, just optimize array dtypes
        if "X_list" in chunk_data:
            optimized_X_list = []
            for X in chunk_data["X_list"]:
                # Convert float64 to float32 if precision allows
                if X.dtype == np.float64:
                    X_float32 = X.astype(np.float32)
                    if np.allclose(X, X_float32, rtol=1e-6):
                        optimized_X_list.append(X_float32)
                        logger.debug("Compressed array from float64 to float32")
                    else:
                        optimized_X_list.append(X)
                else:
                    optimized_X_list.append(X)

            chunk_data["X_list"] = optimized_X_list

        return chunk_data

    def save_chunked_data(
        self,
        data: Dict[str, Any],
        base_path: Union[str, Path],
        chunk_subjects: int = None,
        format: str = "hdf5",
    ) -> List[Path]:
        """
        Save data in chunks to disk.

        Parameters
        ----------
        data : Dict[str, Any]
            Data to save.
        base_path : Union[str, Path]
            Base path for chunk files.
        chunk_subjects : int, optional
            Subjects per chunk.
        format : str
            Storage format ('hdf5', 'pickle').

        Returns
        -------
        List[Path] : Paths to created chunk files.
        """
        base_path = Path(base_path)
        chunk_files = []

        for chunk_id, chunk_data in enumerate(
            self.load_multiview_data_chunked(data, chunk_subjects)
        ):
            if format == "hdf5":
                chunk_file = base_path / f"chunk_{chunk_id:04d}.h5"
                self._save_chunk_hdf5(chunk_data, chunk_file)
            elif format == "pickle":
                chunk_file = base_path / f"chunk_{chunk_id:04d}.pkl"
                self._save_chunk_pickle(chunk_data, chunk_file)
            else:
                raise ValueError(f"Unsupported format: {format}")

            chunk_files.append(chunk_file)
            logger.debug(f"Saved chunk {chunk_id} to {chunk_file}")

        logger.info(f"Saved data in {len(chunk_files)} chunks to {base_path}")
        return chunk_files

    def _save_chunk_hdf5(self, chunk_data: Dict[str, Any], filepath: Path):
        """Save chunk in HDF5 format."""
        with h5py.File(filepath, "w") as f:
            # Save arrays
            for i, X in enumerate(chunk_data["X_list"]):
                f.create_dataset(f"X_view_{i}", data=X, compression="gzip")

            # Save metadata as attributes
            f.attrs["n_subjects"] = chunk_data["X_list"][0].shape[0]
            f.attrs["n_views"] = len(chunk_data["X_list"])

            # Save subject IDs
            f.create_dataset(
                "subject_ids",
                data=[s.encode("utf-8") for s in chunk_data["subject_ids"]],
            )

            # Save clinical data if present
            if "clinical" in chunk_data and chunk_data["clinical"] is not None:
                clinical_group = f.create_group("clinical")
                for col in chunk_data["clinical"].columns:
                    clinical_group.create_dataset(
                        col, data=chunk_data["clinical"][col].values
                    )

    def _save_chunk_pickle(self, chunk_data: Dict[str, Any], filepath: Path):
        """Save chunk in pickle format."""
        with open(filepath, "wb") as f:
            pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_chunked_data(
        self, chunk_files: List[Path], chunk_indices: List[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Load data from chunk files.

        Parameters
        ----------
        chunk_files : List[Path]
            Paths to chunk files.
        chunk_indices : List[int], optional
            Specific chunks to load. If None, loads all.

        Yields
        ------
        Dict[str, Any] : Loaded chunk data.
        """
        if chunk_indices is None:
            chunk_indices = list(range(len(chunk_files)))

        for idx in chunk_indices:
            if idx >= len(chunk_files):
                continue

            chunk_file = chunk_files[idx]

            if chunk_file.suffix == ".h5":
                yield self._load_chunk_hdf5(chunk_file)
            elif chunk_file.suffix == ".pkl":
                yield self._load_chunk_pickle(chunk_file)
            else:
                raise ValueError(f"Unsupported chunk file format: {chunk_file}")

    def _load_chunk_hdf5(self, filepath: Path) -> Dict[str, Any]:
        """Load chunk from HDF5 format."""
        chunk_data = {}

        with h5py.File(filepath, "r") as f:
            # Load arrays
            X_list = []
            view_idx = 0
            while f"X_view_{view_idx}" in f:
                X_list.append(f[f"X_view_{view_idx}"][:])
                view_idx += 1
            chunk_data["X_list"] = X_list

            # Load subject IDs
            if "subject_ids" in f:
                chunk_data["subject_ids"] = [
                    s.decode("utf-8") for s in f["subject_ids"][:]
                ]

            # Load clinical data
            if "clinical" in f:
                clinical_data = {}
                for col_name in f["clinical"].keys():
                    clinical_data[col_name] = f["clinical"][col_name][:]
                chunk_data["clinical"] = pd.DataFrame(clinical_data)

        return chunk_data

    def _load_chunk_pickle(self, filepath: Path) -> Dict[str, Any]:
        """Load chunk from pickle format."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


@contextmanager
def memory_efficient_data_context(memory_limit_gb: float = 4.0):
    """Context manager for memory-efficient data operations."""
    loader = ChunkedDataLoader(memory_limit_gb=memory_limit_gb)
    optimizer = MemoryOptimizer(max_memory_gb=memory_limit_gb)
    optimizer.start_monitoring()

    try:
        yield loader
    finally:
        optimizer.stop_monitoring()
        optimizer.aggressive_cleanup()


def estimate_chunk_memory_usage(
    n_subjects: int, n_features_per_view: List[int], dtype: np.dtype = np.float32
) -> float:
    """
    Estimate memory usage for a data chunk.

    Parameters
    ----------
    n_subjects : int
        Number of subjects in chunk.
    n_features_per_view : List[int]
        Number of features per view.
    dtype : np.dtype
        Data type of arrays.

    Returns
    -------
    float : Estimated memory usage in GB.
    """
    total_elements = n_subjects * sum(n_features_per_view)
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = total_elements * bytes_per_element

    # Add overhead for data structures (approximately 30%)
    total_bytes *= 1.3

    return total_bytes / (1024**3)


def optimize_chunk_size_for_memory(
    total_subjects: int,
    n_features_per_view: List[int],
    available_memory_gb: float,
    dtype: np.dtype = np.float32,
    safety_factor: float = 0.8,
) -> int:
    """
    Calculate optimal chunk size given memory constraints.

    Parameters
    ----------
    total_subjects : int
        Total number of subjects.
    n_features_per_view : List[int]
        Features per view.
    available_memory_gb : float
        Available memory in GB.
    dtype : np.dtype
        Data type.
    safety_factor : float
        Safety factor to avoid memory pressure.

    Returns
    -------
    int : Optimal chunk size (number of subjects).
    """
    # Calculate memory per subject
    features_per_subject = sum(n_features_per_view)
    bytes_per_subject = (
        features_per_subject * np.dtype(dtype).itemsize * 1.3
    )  # 30% overhead

    # Calculate max subjects that fit in memory
    safe_memory_bytes = available_memory_gb * (1024**3) * safety_factor
    max_subjects_per_chunk = int(safe_memory_bytes / bytes_per_subject)

    # Ensure at least 1 subject per chunk
    chunk_size = max(1, min(max_subjects_per_chunk, total_subjects))

    logger.debug(
        f"Optimal chunk size: {chunk_size} subjects "
        f"(memory per chunk: {estimate_chunk_memory_usage(chunk_size, n_features_per_view, dtype):.2f}GB)"
    )

    return chunk_size
