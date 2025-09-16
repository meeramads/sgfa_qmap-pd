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

    def __init__(self, config):
        self.config = config

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

            except Exception as e:
                logger.error(f"Run {run_id} failed: {e}")
                continue

        return results

    def _run_single_mcmc(
        self, X_list: List[np.ndarray], hypers: Dict, run_id: int
    ) -> Dict:
        """Execute a single MCMC run."""
        from models.sparse_gfa import SparseGFAModel

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
