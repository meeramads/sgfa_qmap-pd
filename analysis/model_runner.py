# analysis/model_runner.py
"""Module for running MCMC models."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import jax
import numpy as np
from numpyro.infer import MCMC, NUTS

logger = logging.getLogger(__name__)


@dataclass
class ModelRunnerConfig:
    """Configuration for ModelRunner.

    Attributes:
        num_runs: Number of independent MCMC runs
        num_warmup: Number of warmup iterations
        num_samples: Number of sampling iterations
        num_chains: Number of parallel chains
        model: Model name/identifier
        enable_memory_optimization: Enable automatic memory optimization
        memory_limit_gb: Memory limit in GB for optimization
    """
    num_runs: int = 1
    num_warmup: int = 1000
    num_samples: int = 2000
    num_chains: int = 1
    model: str = "sparseGFA"
    enable_memory_optimization: bool = False
    memory_limit_gb: float = 16.0

    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelRunnerConfig':
        """Create config from dictionary, extracting only valid fields."""
        valid_fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

    @classmethod
    def from_object(cls, obj: Any) -> 'ModelRunnerConfig':
        """Create config from object with attributes."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)

        # Extract attributes from object
        valid_fields = cls.__dataclass_fields__.keys()
        kwargs = {}
        for field_name in valid_fields:
            if hasattr(obj, field_name):
                kwargs[field_name] = getattr(obj, field_name)
        return cls(**kwargs)


class ModelRunner:
    """Handles MCMC model execution."""

    def __init__(self, config: ModelRunnerConfig, results_dir: Optional[Path] = None):
        """Initialize ModelRunner with proper config.

        Args:
            config: ModelRunnerConfig instance
            results_dir: Optional directory for saving results

        Raises:
            TypeError: If config is not a ModelRunnerConfig instance
        """
        if not isinstance(config, ModelRunnerConfig):
            raise TypeError(
                f"ModelRunner requires ModelRunnerConfig, got {type(config).__name__}. "
                f"Use ModelRunnerConfig.from_object(config) or ModelRunnerConfig.from_dict(config) "
                f"to convert your config first."
            )

        self.config = config
        self.results_dir = results_dir
        self.output_dir = results_dir

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

                # Process robust parameters
                if self.config.num_chains > 1:
                    # Multi-chain: extract robust components across chains
                    robust_params = self._extract_robust_parameters(run_results)
                    results[run_id]["robust"] = robust_params
                else:
                    # Single-chain: create robust parameters from the single chain
                    robust_params = self._create_single_chain_robust_parameters(run_results)
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

        # Use memory-optimized execution if enabled
        if self.config.enable_memory_optimization:
            return self._run_single_mcmc_optimized(X_list, hypers, run_id)

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

    def _run_single_mcmc_optimized(
        self, X_list: List[np.ndarray], hypers: Dict, run_id: int
    ) -> Dict:
        """
        Execute a single MCMC run with memory optimization.

        This method uses the memory-optimized execution path through
        core.model_interface.execute_sgfa_model(), which automatically
        adjusts MCMC parameters based on available memory.

        Args:
            X_list: List of data matrices
            hypers: Hyperparameters dictionary
            run_id: Run identifier

        Returns:
            Dictionary with samples and metadata
        """
        from core.model_interface import execute_sgfa_model

        # Prepare MCMC config
        mcmc_config = {
            "num_warmup": self.config.num_warmup,
            "num_samples": self.config.num_samples,
            "num_chains": self.config.num_chains,
            "random_seed": np.random.randint(0, 10000),
            "target_accept_prob": 0.9,
            "max_tree_depth": 12,
        }

        # Prepare model args (convert config to namespace-like object)
        import argparse
        model_args = argparse.Namespace(
            model=self.config.model,
            K=hypers.get("K", 10),
        )

        # Run with memory optimization
        start_time = time.time()
        result = execute_sgfa_model(
            X_list,
            hypers,
            model_args,
            mcmc_config=mcmc_config,
            enable_memory_optimization=True,
            memory_limit_gb=self.config.memory_limit_gb,
        )
        elapsed = time.time() - start_time

        # Extract samples and add metadata
        samples = result["samples"]
        samples["elapsed_time"] = elapsed
        samples["run_id"] = run_id

        # Log optimization info if available
        if "optimization_info" in result:
            opt_info = result["optimization_info"]
            if opt_info.get("memory_optimized"):
                logger.info(
                    f"Run {run_id}: Memory optimization applied. "
                    f"Limit: {opt_info['memory_limit_gb']}GB"
                )

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

    def _create_single_chain_robust_parameters(self, run_results: dict) -> dict:
        """Create robust parameters from a single chain by using posterior means."""
        try:
            # run_results IS the samples dict (returned directly from _run_single_mcmc)
            # Check if we have the expected MCMC sample structure
            if not run_results or not any(key in run_results for key in ["W", "Z", "sigma"]):
                logger.warning("No MCMC samples found in run results for robust parameter creation")
                return {"extraction_successful": False, "reason": "no_mcmc_samples"}

            samples = run_results

            # Extract key parameters and compute posterior means
            robust_params = {}

            # Factor loadings (W) - most important for visualization
            if "W" in samples:
                W = samples["W"]
                if hasattr(W, 'mean') and callable(getattr(W, 'mean', None)):
                    robust_params["W"] = W.mean(axis=0)  # Average across MCMC samples
                else:
                    robust_params["W"] = W
                logger.debug(f"Single-chain robust W shape: {robust_params['W'].shape}")

            # Factor scores (Z)
            if "Z" in samples:
                Z = samples["Z"]
                if hasattr(Z, 'mean') and callable(getattr(Z, 'mean', None)):
                    robust_params["Z"] = Z.mean(axis=0)  # Average across MCMC samples
                else:
                    robust_params["Z"] = Z
                logger.debug(f"Single-chain robust Z shape: {robust_params['Z'].shape}")

            # Additional parameters if available
            for param_name in ["sigma", "tauZ", "lmbZ", "cZ", "tauW", "lmbW", "cW"]:
                if param_name in samples:
                    param_value = samples[param_name]
                    if hasattr(param_value, 'mean') and callable(getattr(param_value, 'mean', None)):
                        robust_params[param_name] = param_value.mean(axis=0)
                    else:
                        robust_params[param_name] = param_value

            # Reconstruction (if needed for visualization)
            if "W" in robust_params and "Z" in robust_params:
                # Compute data reconstruction X = Z @ W.T
                Z_robust = robust_params["Z"]
                W_robust = robust_params["W"]
                try:
                    infX = Z_robust @ W_robust.T
                    robust_params["infX"] = infX
                    logger.debug(f"Single-chain reconstruction shape: {infX.shape}")
                except Exception as e:
                    logger.warning(f"Could not compute reconstruction: {e}")

            robust_params.update({
                "extraction_successful": True,
                "n_robust_components": robust_params["W"].shape[1] if "W" in robust_params else 0,
                "single_chain": True,
                "method": "posterior_mean"
            })

            logger.info(f"✅ Created single-chain robust parameters with {robust_params['n_robust_components']} components")
            return robust_params

        except Exception as e:
            logger.error(f"Single-chain robust parameter creation failed: {e}")
            return {
                "extraction_successful": False,
                "reason": f"creation_error: {str(e)}",
                "single_chain": True
            }

    def _save_run_results(self, run_results: dict, run_id: int, data: dict = None):
        """Save results immediately after each successful run."""
        try:
            from core.utils import safe_pickle_save
            from pathlib import Path

            # Create output directory if it doesn't exist
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save model parameters (disabled to prevent disk quota issues)
            # model_params_file = output_dir / f"[{run_id}]Model_params.dictionary"
            # safe_pickle_save(run_results, model_params_file, f"Model parameters run {run_id}")

            # Save robust parameters if they exist (disabled to prevent disk quota issues)
            # if "robust" in run_results:
            #     robust_params_file = output_dir / f"[{run_id}]Robust_params.dictionary"
            #     safe_pickle_save(run_results["robust"], robust_params_file, f"Robust parameters run {run_id}")

            logger.info(f"Dictionary file saving disabled to prevent disk quota issues for run {run_id}")

            # Save factor loadings for immediate access
            # run_results IS the samples dict (returned directly from _run_single_mcmc)
            samples = run_results
            if "W" in samples:
                import numpy as np

                # Extract factor loadings
                W = samples["W"]
                if hasattr(W, 'mean'):  # If it's MCMC samples, take mean
                    W = W.mean(axis=0)

                # Save factor loadings as CSV file for easy access
                W_csv_file = output_dir / f"[{run_id}]Factor_loadings_W.csv"
                import pandas as pd
                W_df = pd.DataFrame(W, columns=[f"Factor_{i+1}" for i in range(W.shape[1])])
                W_df.index.name = "Feature_Index"
                W_df.to_csv(W_csv_file, index=True)
                logger.info(f"Saved factor loadings CSV: {W_csv_file}")

                # Save weight vectors in concatenated volume-matrix format (like original data)
                self._save_concatenated_weight_vectors(W, output_dir, run_id)

            if "Z" in samples:
                # Extract factor scores
                Z = samples["Z"]
                if hasattr(Z, 'mean'):  # If it's MCMC samples, take mean
                    Z = Z.mean(axis=0)

                # Save factor scores as CSV file for easy access
                Z_csv_file = output_dir / f"[{run_id}]Factor_scores_Z.csv"

                # Log the shape for debugging
                logger.info(f"Factor scores Z shape: {Z.shape} (should be {data.get('n_subjects', 'unknown')} subjects x {Z.shape[1]} factors)")

                # Create DataFrame with proper subject IDs
                Z_df = pd.DataFrame(Z, columns=[f"Factor_{i+1}" for i in range(Z.shape[1])])
                Z_df.index = [f"Subject_{i+1}" for i in range(Z.shape[0])]  # Start from Subject_1, not 0
                Z_df.index.name = "Subject_ID"
                Z_df.to_csv(Z_csv_file, index=True)
                logger.info(f"Saved factor scores CSV: {Z_csv_file} ({Z.shape[0]} subjects, {Z.shape[1]} factors)")

            logger.info(f"💾 Run {run_id} results saved to {output_dir}")

        except Exception as e:
            logger.error(f"Failed to save run {run_id} results: {e}")
            # Don't fail the entire run just because saving failed

    def _save_concatenated_weight_vectors(self, W: np.ndarray, output_dir: Path, run_id: int):
        """Save weight vectors in concatenated format similar to original volume matrices.

        This creates separate files for each brain region (like volume_sn_voxels.tsv)
        with weights organized as: rows = factors, columns = voxels within that region.
        """
        try:
            import numpy as np
            import pandas as pd
            from pathlib import Path

            logger.info(f"💾 Saving concatenated weight vectors for run {run_id}...")

            # Load position files to determine voxel counts for each region
            data_dir = Path("qMAP-PD_data")
            position_files = {
                "sn": data_dir / "position_lookup" / "position_sn_voxels.tsv",
                "putamen": data_dir / "position_lookup" / "position_putamen_voxels.tsv",
                "lentiform": data_dir / "position_lookup" / "position_lentiform_voxels.tsv"
            }

            # Get dimensions for each region
            region_dims = {}
            feature_start = 0

            for region_name, pos_file in position_files.items():
                if pos_file.exists():
                    positions = pd.read_csv(pos_file, sep="\t", header=None).values.flatten()
                    region_dim = len(positions)
                    region_dims[region_name] = {
                        "start": feature_start,
                        "end": feature_start + region_dim,
                        "dim": region_dim
                    }
                    feature_start += region_dim
                    logger.debug(f"Region {region_name}: {region_dim} voxels (features {region_dims[region_name]['start']}-{region_dims[region_name]['end']-1})")
                else:
                    logger.warning(f"Position file not found: {pos_file}")

            # Extract and save weight vectors for each brain region
            for region_name, dims in region_dims.items():
                # Extract weights for this region: W[start:end, :] -> (region_voxels, n_factors)
                region_weights = W[dims["start"]:dims["end"], :]

                # Transpose to match volume matrix format: (n_factors, region_voxels)
                # This makes each row a factor and each column a voxel (like subjects x voxels in original)
                region_weights_transposed = region_weights.T

                # Save as TSV file similar to volume_sn_voxels.tsv format
                weight_file = output_dir / f"[{run_id}]weights_{region_name}_voxels.tsv"
                np.savetxt(weight_file, region_weights_transposed, delimiter='\t', fmt='%.6f')

                logger.info(f"✅ Saved {region_name} weights: {weight_file} ({region_weights_transposed.shape[0]} factors × {region_weights_transposed.shape[1]} voxels)")

            # Also save clinical weights separately
            clinical_file = data_dir / "data_clinical" / "pd_motor_gfa_data.tsv"
            if clinical_file.exists():
                clinical_df = pd.read_csv(clinical_file, sep="\t")
                clinical_features = clinical_df.shape[1] - 1  # Minus ID column

                if feature_start + clinical_features <= W.shape[0]:
                    # Extract clinical weights
                    clinical_weights = W[feature_start:feature_start + clinical_features, :]
                    clinical_weights_transposed = clinical_weights.T  # (n_factors, n_clinical_vars)

                    # Save clinical weights with variable names as column headers
                    clinical_names = [col for col in clinical_df.columns if col != 'sid']
                    clinical_weight_file = output_dir / f"[{run_id}]weights_clinical_variables.tsv"

                    # Create DataFrame with proper column names
                    clinical_df_weights = pd.DataFrame(
                        clinical_weights_transposed,
                        columns=clinical_names,
                        index=[f"Factor_{i+1}" for i in range(clinical_weights_transposed.shape[0])]
                    )
                    clinical_df_weights.to_csv(clinical_weight_file, sep='\t', float_format='%.6f')

                    logger.info(f"✅ Saved clinical weights: {clinical_weight_file} ({clinical_weights_transposed.shape[0]} factors × {len(clinical_names)} variables)")
                else:
                    logger.warning(f"Weight matrix too small for expected clinical features: {W.shape[0]} < {feature_start + clinical_features}")

            # Create a summary file explaining the format
            summary_file = output_dir / f"[{run_id}]weight_vectors_README.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Weight Vector Storage Format (Run {run_id})\n")
                f.write("=" * 50 + "\n\n")
                f.write("This directory contains factor loadings (weight vectors) stored in a format\n")
                f.write("similar to the original volume matrices for easy reconstruction:\n\n")

                f.write("Brain Region Files:\n")
                for region_name, dims in region_dims.items():
                    f.write(f"  [{run_id}]weights_{region_name}_voxels.tsv\n")
                    f.write(f"    - Format: {W.shape[1]} factors × {dims['dim']} voxels\n")
                    f.write(f"    - Each row = one factor's weights for {region_name} voxels\n")
                    f.write(f"    - Each column = one voxel's weights across all factors\n")
                    f.write(f"    - Voxel positions defined in: position_{region_name}_voxels.tsv\n\n")

                f.write("Clinical Variables File:\n")
                f.write(f"  [{run_id}]weights_clinical_variables.tsv\n")
                f.write(f"    - Format: {W.shape[1]} factors × {clinical_features if 'clinical_features' in locals() else 'N'} clinical variables\n")
                f.write(f"    - Each row = one factor's weights for clinical variables\n")
                f.write(f"    - Column names = actual clinical variable names (age, sex, etc.)\n\n")

                f.write("Usage:\n")
                f.write("  - Use position files to map voxel weights back to 3D brain coordinates\n")
                f.write("  - Combine with original volume matrix structure for brain reconstruction\n")
                f.write("  - Clinical weights show factor relationships to demographic/clinical measures\n")

            logger.info(f"✅ Saved weight vector summary: {summary_file}")
            logger.info(f"💾 Concatenated weight vector storage completed for run {run_id}")

        except Exception as e:
            logger.error(f"Failed to save concatenated weight vectors for run {run_id}: {e}")
            # Don't fail the entire run just because this additional saving failed
