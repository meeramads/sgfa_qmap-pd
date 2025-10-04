"""Results cache for sharing SGFA analysis results across experiments."""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SGFAResultsCache:
    """Cache for SGFA analysis results to avoid redundant computation across experiments."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the results cache.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory to store cached results. If None, uses in-memory cache only.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.memory_cache: Dict[str, Dict] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📦 SGFA results cache initialized: {self.cache_dir}")
        else:
            logger.info("📦 SGFA results cache initialized (memory-only)")

    def _compute_hash(self, X_list: List[np.ndarray], hypers: Dict, args: Any) -> str:
        """
        Compute a hash key for the SGFA run based on data and parameters.

        Parameters
        ----------
        X_list : List[np.ndarray]
            Data matrices
        hypers : Dict
            Hyperparameters
        args : Any
            Additional arguments (config object or dict)

        Returns
        -------
        str
            Hash key for this configuration
        """
        # Extract relevant parameters for hashing
        hash_dict = {
            "data_shapes": [X.shape for X in X_list],
            "data_checksums": [hashlib.md5(X.tobytes()).hexdigest()[:8] for X in X_list],
            "hypers": {k: v for k, v in sorted(hypers.items())},
        }

        # Extract relevant args
        if hasattr(args, "__dict__"):
            args_dict = {
                "num_warmup": getattr(args, "num_warmup", None),
                "num_samples": getattr(args, "num_samples", None),
                "num_chains": getattr(args, "num_chains", None),
                "seed": getattr(args, "seed", None),
            }
        else:
            args_dict = {
                "num_warmup": args.get("num_warmup"),
                "num_samples": args.get("num_samples"),
                "num_chains": args.get("num_chains"),
                "seed": args.get("seed"),
            }

        hash_dict["args"] = {k: v for k, v in sorted(args_dict.items()) if v is not None}

        # Compute hash
        hash_str = json.dumps(hash_dict, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]

    def get(
        self, X_list: List[np.ndarray], hypers: Dict, args: Any
    ) -> Optional[Dict]:
        """
        Get cached SGFA results if available.

        Parameters
        ----------
        X_list : List[np.ndarray]
            Data matrices
        hypers : Dict
            Hyperparameters
        args : Any
            Additional arguments

        Returns
        -------
        Dict or None
            Cached results if available, None otherwise
        """
        cache_key = self._compute_hash(X_list, hypers, args)

        # Check memory cache first
        if cache_key in self.memory_cache:
            logger.info(f"✅ Cache HIT (memory): {cache_key}")
            return self.memory_cache[cache_key]

        # Check disk cache if available
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        result = pickle.load(f)
                    # Also store in memory cache
                    self.memory_cache[cache_key] = result
                    logger.info(f"✅ Cache HIT (disk): {cache_key}")
                    return result
                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")

        logger.debug(f"❌ Cache MISS: {cache_key}")
        return None

    def put(
        self, X_list: List[np.ndarray], hypers: Dict, args: Any, result: Dict
    ) -> None:
        """
        Store SGFA results in cache.

        Parameters
        ----------
        X_list : List[np.ndarray]
            Data matrices
        hypers : Dict
            Hyperparameters
        args : Any
            Additional arguments
        result : Dict
            SGFA analysis results to cache
        """
        cache_key = self._compute_hash(X_list, hypers, args)

        # Store in memory cache
        self.memory_cache[cache_key] = result

        # Store in disk cache if available
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
                logger.info(f"💾 Cached results: {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to save cache file {cache_file}: {e}")

    def clear(self) -> None:
        """Clear all cached results."""
        self.memory_cache.clear()

        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("🗑️  Cache cleared")

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns
        -------
        Dict[str, int]
            Cache statistics (memory_entries, disk_entries)
        """
        stats = {"memory_entries": len(self.memory_cache)}

        if self.cache_dir and self.cache_dir.exists():
            stats["disk_entries"] = len(list(self.cache_dir.glob("*.pkl")))
        else:
            stats["disk_entries"] = 0

        return stats
