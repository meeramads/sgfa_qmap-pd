# analysis/model_runner.py
"""Module for running MCMC models."""

import logging
import time
from typing import Dict, List

import jax
import numpy as np
from numpyro.infer import MCMC, NUTS

logger = logging.getLogger(__name__)


class ModelRunner:
    """Handles MCMC model execution."""

    def __init__(self, config_or_args, results_dir=None):
        # Support both old (config) and new (args, results_dir) patterns
        if results_dir is not None:
            # New pattern: ModelRunner(args, results_dir)
            self.args = config_or_args
            self.results_dir = results_dir
            self.output_dir = results_dir  # Fix: Add output_dir attribute
            # For backward compatibility
            self.config = config_or_args
        else:
            # Old pattern: ModelRunner(config)
            self.config = config_or_args
            self.args = config_or_args
            self.results_dir = None
            self.output_dir = None  # Fix: Add output_dir attribute

    def run_standard_analysis(
        self, X_list: List[np.ndarray], hypers: Dict, data: Dict
    ) -> Dict:
        """Run standard MCMC analysis."""
        results = {}

        for run_id in range(1, self.config.num_runs + 1):
            logger.info(f"Starting run {run_id}/{self.config.num_runs}")

            try:
                run_results = self._run_single_mcmc(X_list, hypers, run_id)
                results[run_id] = run_results

                # Process robust parameters if multi-chain
                if self.config.num_chains > 1:
                    robust_params = self._extract_robust_parameters(run_results)
                    results[run_id]["robust"] = robust_params

                # CRITICAL: Save results immediately after each successful run
                self._save_run_results(run_results, run_id, data if 'data' in locals() else None)
                logger.info(f"✅ Run {run_id} completed and saved successfully")

            except Exception as e:
                logger.error(f"Run {run_id} failed: {e}")
                continue

            finally:
                # Critical: GPU memory cleanup after each run
                self._cleanup_run_memory(run_id)

        return results

    def _run_single_mcmc(
        self, X_list: List[np.ndarray], hypers: Dict, run_id: int
    ) -> Dict:
        """Execute a single MCMC run."""
        from models.sparse_gfa import SparseGFAModel

        # Aggressive memory cleanup before starting run (from experiments)
        if run_id > 1:  # Skip for first run
            self._pre_run_memory_cleanup(run_id)

        # Initialize model
        model = SparseGFAModel(self.config, hypers)

        # Setup MCMC
        rng_key = jax.random.PRNGKey(np.random.randint(0, 10000))
        kernel = NUTS(model, target_accept_prob=0.9, max_tree_depth=12)
        mcmc = MCMC(
            kernel,
            num_warmup=self.config.num_warmup,
            num_samples=self.config.num_samples,
            num_chains=self.config.num_chains,
        )

        # Run inference
        start_time = time.time()
        mcmc.run(rng_key, X_list)
        elapsed = time.time() - start_time

        # Get samples
        samples = mcmc.get_samples()
        samples["elapsed_time"] = elapsed
        samples["run_id"] = run_id

        return samples

    def _extract_robust_parameters(self, run_results: Dict) -> Dict:
        """Extract robust parameters from multi-chain results."""
        # Use existing utilities instead of non-existent analysis.robust_extraction
        from core.utils import get_infparams, get_robustK

        try:
            # Get hyperparameters from config
            hypers = getattr(self.config, "hypers", {})

            # Extract inferred parameters using existing utils
            inf_params, data_comps = get_infparams(run_results, hypers, self.config)

            if self.config.num_chains > 1:
                # Find robust components across chains
                thresholds = {"cosineThr": 0.8, "matchThr": 0.5}
                robust_params, X_robust, success = get_robustK(
                    thresholds, self.config, inf_params, data_comps
                )

                if success:
                    # Add metadata
                    robust_params.update(
                        {
                            "sigma_inf": inf_params.get("sigma"),
                            "infX": X_robust,
                            "extraction_successful": True,
                            "n_robust_components": (
                                robust_params.get("W", np.array([])).shape[1]
                                if "W" in robust_params
                                else 0
                            ),
                        }
                    )

                    # Add model-specific parameters
                    if (
                        hasattr(self.config, "model")
                        and "sparseGFA" in self.config.model
                    ):
                        if "tauW" in inf_params:
                            robust_params["tauW_inf"] = inf_params["tauW"]

                    return robust_params
                else:
                    logger.warning("No robust components found across chains")
                    return {
                        "extraction_successful": False,
                        "reason": "no_robust_components",
                    }
            else:
                # Single chain case - use mean parameters
                W = np.mean(inf_params["W"][0], axis=0) if "W" in inf_params else None
                Z = np.mean(inf_params["Z"][0], axis=0) if "Z" in inf_params else None

                if W is not None and Z is not None:
                    X_recon = np.dot(Z, W.T)

                    robust_params = {
                        "W": W,
                        "Z": Z,
                        "infX": X_recon,
                        "sigma_inf": inf_params.get("sigma"),
                        "extraction_successful": True,
                        "n_robust_components": W.shape[1],
                        "single_chain": True,
                    }

                    return robust_params
                else:
                    return {
                        "extraction_successful": False,
                        "reason": "missing_parameters",
                    }

        except Exception as e:
            logger.error(f"Robust parameter extraction failed: {e}")
            return {
                "extraction_successful": False,
                "reason": f"extraction_error: {str(e)}",
            }

    def _cleanup_run_memory(self, run_id: int):
        """Aggressive GPU memory cleanup after each MCMC run using experiment techniques."""
        try:
            logger.info(f"🧹 Performing aggressive memory cleanup after run {run_id}")
            import gc
            import jax

            # More aggressive JAX cache clearing (from experiments)
            try:
                from jax._src import compilation_cache
                compilation_cache.clear_cache()
                logger.debug("JAX compilation cache cleared")
            except Exception as e:
                logger.debug(f"Could not clear JAX cache: {e}")

            # Clear all JAX caches (multiple methods)
            try:
                jax.clear_caches()
                logger.debug("JAX caches cleared")
            except Exception as e:
                logger.debug(f"Could not clear general JAX caches: {e}")

            # Force multiple garbage collection cycles (from experiments)
            collected_objects = 0
            for i in range(5):  # More aggressive: 5 cycles instead of 3
                collected = gc.collect()
                collected_objects += collected
            logger.debug(f"Garbage collection freed {collected_objects} objects")

            # More aggressive GPU memory cleanup
            try:
                for device in jax.devices():
                    if device.platform == 'gpu':
                        # Multiple cleanup methods for GPU
                        try:
                            device.synchronize_all_activity()
                        except Exception:
                            pass
                        try:
                            device.memory_stats()  # Force memory stats refresh
                        except Exception:
                            pass
                logger.debug("GPU memory cleanup attempted")
            except Exception as e:
                logger.debug(f"GPU memory cleanup failed: {e}")

            # Additional cleanup: explicitly delete large variables from memory
            try:
                # Force deletion of any large arrays that might be lingering
                import sys
                frame = sys._getframe(1)  # Get calling frame
                local_vars = list(frame.f_locals.keys())
                for var_name in local_vars:
                    if var_name.startswith(('samples', 'mcmc', 'results', 'X_', 'W', 'Z')):
                        try:
                            del frame.f_locals[var_name]
                        except Exception:
                            pass
            except Exception:
                pass

            # Longer delay for cleanup to complete (from experiments)
            import time
            time.sleep(2)  # Increased from 1 to 2 seconds

            logger.info(f"✅ Aggressive memory cleanup completed for run {run_id}")

        except Exception as e:
            logger.warning(f"Memory cleanup after run {run_id} failed: {e}")

    def _pre_run_memory_cleanup(self, run_id: int):
        """Aggressive memory cleanup before starting a run (from experiment techniques)."""
        try:
            logger.info(f"🧹 Pre-run aggressive memory cleanup before run {run_id}")
            import gc
            import jax

            # Clear JAX compilation cache (from experiments)
            try:
                from jax._src import compilation_cache
                compilation_cache.clear_cache()
                logger.debug("Pre-run: JAX compilation cache cleared")
            except Exception:
                pass

            # Clear all JAX caches
            try:
                jax.clear_caches()
                logger.debug("Pre-run: JAX caches cleared")
            except Exception:
                pass

            # Multiple garbage collection cycles (from experiments)
            for i in range(5):
                gc.collect()

            # GPU memory synchronization and cleanup
            try:
                for device in jax.devices():
                    if device.platform == 'gpu':
                        device.synchronize_all_activity()
                        device.memory_stats()
                logger.debug("Pre-run: GPU memory synchronized")
            except Exception:
                pass

            # Brief delay for cleanup (from experiments)
            import time
            time.sleep(2)

            logger.debug(f"✅ Pre-run cleanup completed for run {run_id}")

        except Exception as e:
            logger.warning(f"Pre-run memory cleanup before run {run_id} failed: {e}")

    def _save_run_results(self, run_results: dict, run_id: int, data: dict = None):
        """Save results immediately after each successful run."""
        try:
            from core.utils import safe_pickle_save
            from pathlib import Path

            # Create output directory if it doesn't exist
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save model parameters
            model_params_file = output_dir / f"[{run_id}]Model_params.dictionary"
            safe_pickle_save(run_results, model_params_file, f"Model parameters run {run_id}")

            # Save robust parameters if they exist
            if "robust" in run_results:
                robust_params_file = output_dir / f"[{run_id}]Robust_params.dictionary"
                safe_pickle_save(run_results["robust"], robust_params_file, f"Robust parameters run {run_id}")

            # Save factor loadings for immediate access
            if "samples" in run_results:
                samples = run_results["samples"]
                if "W" in samples:
                    import numpy as np

                    # Save factor loadings as numpy array for easy access
                    W_file = output_dir / f"[{run_id}]Factor_loadings_W.npy"
                    W = samples["W"]
                    if hasattr(W, 'mean'):  # If it's MCMC samples, take mean
                        W = W.mean(axis=0)
                    np.save(W_file, W)

                if "Z" in samples:
                    # Save factor scores
                    Z_file = output_dir / f"[{run_id}]Factor_scores_Z.npy"
                    Z = samples["Z"]
                    if hasattr(Z, 'mean'):  # If it's MCMC samples, take mean
                        Z = Z.mean(axis=0)
                    np.save(Z_file, Z)

            logger.info(f"💾 Run {run_id} results saved to {output_dir}")

        except Exception as e:
            logger.error(f"Failed to save run {run_id} results: {e}")
            # Don't fail the entire run just because saving failed
