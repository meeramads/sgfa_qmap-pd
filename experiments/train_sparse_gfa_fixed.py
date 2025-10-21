#!/usr/bin/env python3
"""Training script for SGFA with fixed parameters and factor stability analysis.

This script implements a streamlined SGFA training pipeline following
Ferreira et al. 2024 methodology, focusing on factor stability assessment
rather than hyperparameter exploration.

Usage:
    python experiments/train_sparse_gfa_fixed.py --config config.yaml --K 20

Key Features:
- Fixed hyperparameters (no grid search)
- 4 independent MCMC chains (sequential execution)
- Factor stability analysis using cosine similarity
- Effective factor counting
- Memory-efficient sequential execution
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config_utils import get_data_dir, get_output_dir
from core.logger_utils import setup_logger
from experiments.framework import ExperimentConfig, ExperimentFramework
from experiments.robustness_testing import RobustnessExperiments

# Fixed configuration based on Ferreira et al. 2024
FIXED_CONFIG = {
    # Model structure
    "K": 20,  # Number of latent factors (will shrink to fewer)
    "model_type": "sparse_gfa",

    # MCMC settings
    "num_chains": 4,  # Run 4 independent chains
    "num_warmup": 1000,  # Warmup samples
    "num_samples": 5000,  # Posterior samples

    # Sparsity priors (Section 2.7 of Ferreira et al. 2024)
    "percW": 33,  # p₀^(m) = Dm/3 (33% sparsity)
    "slab_df": 4,  # ν = 4 (slab degrees of freedom)
    "slab_scale": 2,  # s = 2 (slab scale)

    # Prior hyperparameters
    "aρ": 1,  # Shape parameter for noise prior
    "bρ": 1,  # Rate parameter for noise prior

    # Other settings
    "reghsZ": True,  # Regularized horseshoe on Z
    "target_accept_prob": 0.8,
    "random_seed": 42,
}

# Factor stability parameters
STABILITY_CONFIG = {
    "n_chains": 4,  # Number of independent chains
    "cosine_threshold": 0.8,  # Minimum similarity for matching
    "min_match_rate": 0.5,  # Minimum match rate for robust factor (>50%)
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SGFA with fixed parameters and assess factor stability"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--K",
        type=int,
        default=FIXED_CONFIG["K"],
        help=f"Number of latent factors (default: {FIXED_CONFIG['K']})",
    )

    parser.add_argument(
        "--num-chains",
        type=int,
        default=STABILITY_CONFIG["n_chains"],
        help=f"Number of independent chains (default: {STABILITY_CONFIG['n_chains']})",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=FIXED_CONFIG["num_samples"],
        help=f"Number of MCMC samples (default: {FIXED_CONFIG['num_samples']})",
    )

    parser.add_argument(
        "--num-warmup",
        type=int,
        default=FIXED_CONFIG["num_warmup"],
        help=f"Number of warmup samples (default: {FIXED_CONFIG['num_warmup']})",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: from config)",
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save stability analysis results to files",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_hyperparameters(X_list: List[np.ndarray], K: int) -> Dict:
    """Prepare hyperparameters for SGFA model.

    Parameters
    ----------
    X_list : List[np.ndarray]
        List of data matrices
    K : int
        Number of latent factors

    Returns
    -------
    Dict
        Hyperparameters for SGFA model
    """
    hypers = {
        "Dm": [X.shape[1] for X in X_list],
        "a_sigma": FIXED_CONFIG["aρ"],
        "b_sigma": FIXED_CONFIG["bρ"],
        "slab_df": FIXED_CONFIG["slab_df"],
        "slab_scale": FIXED_CONFIG["slab_scale"],
        "percW": FIXED_CONFIG["percW"],
        "K": K,
    }
    return hypers


def prepare_mcmc_args(args) -> Dict:
    """Prepare MCMC arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    Dict
        MCMC configuration
    """
    mcmc_args = {
        "K": args.K,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "num_chains": 1,  # Will be set to 1 for sequential execution
        "target_accept_prob": FIXED_CONFIG["target_accept_prob"],
        "reghsZ": FIXED_CONFIG["reghsZ"],
        "random_seed": FIXED_CONFIG["random_seed"],
    }
    return mcmc_args


def main():
    """Main training and analysis pipeline."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("train_sparse_gfa_fixed", level=log_level)

    logger.info("=" * 80)
    logger.info("SGFA Factor Stability Analysis - Fixed Parameters")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  - K (latent factors): {args.K}")
    logger.info(f"  - Number of chains: {args.num_chains}")
    logger.info(f"  - MCMC samples: {args.num_samples}")
    logger.info(f"  - Warmup samples: {args.num_warmup}")
    logger.info(f"  - Cosine threshold: {STABILITY_CONFIG['cosine_threshold']}")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Load data
        logger.info("Loading data...")
        from data.preprocessing_integration import apply_preprocessing_to_pipeline

        X_list, preprocessing_info = apply_preprocessing_to_pipeline(
            config=config,
            data_dir=get_data_dir(config),
            auto_select_strategy=False,
            preferred_strategy=config.get("preprocessing", {}).get("strategy", "standard"),
        )

        logger.info(f"Data loaded: {len(X_list)} views")
        for i, X in enumerate(X_list):
            logger.info(f"  View {i}: {X.shape}")

        # Extract subject_ids, feature_names, and view_names from preprocessing_info
        subject_ids = preprocessing_info.get("data_summary", {}).get("original_data", {}).get("subject_ids", None)
        feature_names_raw = preprocessing_info.get("data_summary", {}).get("original_data", {}).get("feature_names", {})
        view_names = preprocessing_info.get("data_summary", {}).get("view_names", [f"view_{i}" for i in range(len(X_list))])

        # Build feature_names to match actual filtered dimensions
        # Imaging data has no feature names (use generic labels based on filtered voxel count)
        # Clinical data has feature names from headers (after confound removal)
        feature_names = {}
        if feature_names_raw and view_names:
            for view_idx, view_name in enumerate(view_names):
                if view_idx < len(X_list):
                    actual_dim = X_list[view_idx].shape[1]
                    if view_name in feature_names_raw:
                        # Clinical view: use actual feature names from header (already filtered)
                        original_names = feature_names_raw[view_name]
                        if len(original_names) == actual_dim:
                            feature_names[view_name] = original_names
                        else:
                            # Mismatch - use generic labels
                            feature_names[view_name] = [f"{view_name}_feature_{j}" for j in range(actual_dim)]
                    else:
                        # Imaging view: no names, use generic labels
                        feature_names[view_name] = [f"{view_name}_voxel_{j}" for j in range(actual_dim)]

        if subject_ids:
            logger.info(f"  Subject IDs: {len(subject_ids)} patients")
        if feature_names:
            logger.info(f"  Feature names available for {len(feature_names)} views")

        # Prepare hyperparameters and args
        hypers = prepare_hyperparameters(X_list, args.K)
        mcmc_args = prepare_mcmc_args(args)

        logger.info("\nHyperparameters:")
        for key, value in hypers.items():
            logger.info(f"  {key}: {value}")

        # Create experiment framework first (before creating output directory)
        exp_config = ExperimentConfig(
            experiment_name="factor_stability",
            description=f"Factor stability analysis with K={args.K} (fixed parameters)",
            dataset="qmap_pd",
            data_dir=get_data_dir(config),
            K_values=[args.K],
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            model_type="sparse_gfa_fixed",
            K=args.K,
            percW=args.percW if hasattr(args, 'percW') else 33,
            slab_df=4,  # Default from config
            slab_scale=2,  # Default from config
            qc_outlier_threshold=args.qc_outlier_threshold if hasattr(args, 'qc_outlier_threshold') else 3.0,
            max_tree_depth=args.max_tree_depth if hasattr(args, 'max_tree_depth') else 10,
        )

        # Initialize robustness experiments
        repro_exp = RobustnessExperiments(exp_config, logger)

        # Setup output directory - create experiment subdirectory inside the run directory
        # The run directory is already created by ExperimentFramework at repro_exp.base_output_dir
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Create experiment subdirectory inside the run directory
            # Generate semantic experiment name
            semantic_name = repro_exp._generate_semantic_experiment_name("factor_stability", exp_config)
            # Create subdirectory inside base_output_dir (the run directory)
            output_dir = repro_exp.base_output_dir / semantic_name
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nFactor stability output directory: {output_dir}")
        logger.info(f"Run directory: {repro_exp.base_output_dir}")

        # Run factor stability analysis
        logger.info("\n" + "=" * 80)
        logger.info("Running factor stability analysis...")
        logger.info("=" * 80 + "\n")

        result = repro_exp.run_factor_stability_analysis(
            X_list=X_list,
            hypers=hypers,
            args=mcmc_args,
            n_chains=args.num_chains,
            cosine_threshold=STABILITY_CONFIG["cosine_threshold"],
            min_match_rate=STABILITY_CONFIG["min_match_rate"],
            output_dir=str(output_dir),  # Pass experiment-specific output directory
            subject_ids=subject_ids,  # Pass subject IDs for Z score indexing
            view_names=view_names,  # Pass view names for plot labeling
        )

        # Extract results
        stability_results = result.model_results["stability_results"]
        effective_factors = result.model_results["effective_factors"]
        chain_results = result.model_results["chain_results"]

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("FACTOR STABILITY ANALYSIS RESULTS")
        logger.info("=" * 80)

        logger.info("\nStability Summary:")
        logger.info(f"  Total factors (K): {stability_results['total_factors']}")
        logger.info(f"  Robust factors: {stability_results['n_stable_factors']}")
        logger.info(f"  Stability rate: {stability_results['stability_rate']:.1%}")
        logger.info(f"  Robust factor indices: {stability_results['stable_factor_indices']}")

        logger.info("\nEffective Factors Per Chain:")
        for ef in effective_factors:
            logger.info(
                f"  Chain {ef['chain_id']}: {ef['n_effective']}/{ef['total_factors']} "
                f"effective (shrinkage: {ef['shrinkage_rate']:.1%})"
            )

        mean_effective = np.mean([ef["n_effective"] for ef in effective_factors])
        logger.info(f"  Mean effective factors: {mean_effective:.1f}")

        logger.info("\nChain Convergence:")
        for i, chain in enumerate(chain_results):
            logger.info(
                f"  Chain {i}: LL={chain['log_likelihood']:.2f}, "
                f"converged={chain['convergence']}"
            )

        # Save results if requested
        if args.save_results:
            logger.info(f"\nSaving results to {output_dir}...")

            from analysis.factor_stability import save_stability_results

            # Save stability analysis results
            stability_dir = output_dir / "stability_analysis"
            save_stability_results(
                stability_results,
                effective_factors[0],  # Use first chain for effective factor stats
                str(stability_dir),
                subject_ids=subject_ids,  # Pass subject IDs for Z score indexing
                view_names=view_names,  # Pass view names for W indexing
                feature_names=feature_names,  # Pass feature names for W indexing
                Dm=hypers['Dm'],  # Pass view dimensions for per-view consensus saving
            )

            # Save chain results
            chains_dir = output_dir / "chains"
            chains_dir.mkdir(exist_ok=True)

            from core.io_utils import save_csv, save_json
            import pandas as pd

            for i, chain in enumerate(chain_results):
                chain_dir = chains_dir / f"chain_{i}"
                chain_dir.mkdir(exist_ok=True)

                # Save W (factor loadings)
                W = chain["W"]
                if isinstance(W, list):
                    # Multi-view: save each view separately
                    for view_idx, W_view in enumerate(W):
                        W_df = pd.DataFrame(
                            W_view,
                            columns=[f"Factor_{k}" for k in range(W_view.shape[1])],
                        )
                        # Use feature names if available for this view
                        view_name = view_names[view_idx] if view_idx < len(view_names) else f"view_{view_idx}"
                        if feature_names and view_name in feature_names:
                            view_feature_names = feature_names[view_name]
                            if len(view_feature_names) == W_view.shape[0]:
                                W_df.index = view_feature_names
                            else:
                                W_df.index = [f"Feature_{j}" for j in range(W_view.shape[0])]
                        else:
                            W_df.index = [f"Feature_{j}" for j in range(W_view.shape[0])]
                        W_df.index.name = "Feature"
                        save_csv(W_df, chain_dir / f"W_view_{view_idx}.csv", index=True)
                else:
                    # Single matrix
                    W_df = pd.DataFrame(
                        W,
                        columns=[f"Factor_{k}" for k in range(W.shape[1])],
                    )
                    W_df.index = [f"Feature_{j}" for j in range(W.shape[0])]
                    W_df.index.name = "Feature"
                    save_csv(W_df, chain_dir / "W.csv", index=True)

                # Save Z (factor scores)
                Z = chain["Z"]
                Z_df = pd.DataFrame(
                    Z,
                    columns=[f"Factor_{k}" for k in range(Z.shape[1])],
                )
                # Use patient IDs if available, otherwise use generic subject labels
                if subject_ids and len(subject_ids) == Z.shape[0]:
                    Z_df.index = subject_ids
                else:
                    Z_df.index = [f"Subject_{j}" for j in range(Z.shape[0])]
                Z_df.index.name = "Patient ID"
                save_csv(Z_df, chain_dir / "Z.csv", index=True)

                # Save metadata
                metadata = {
                    "chain_id": chain["chain_id"],
                    "seed": chain["seed"],
                    "log_likelihood": float(chain["log_likelihood"]),
                    "convergence": chain["convergence"],
                    "execution_time": chain["execution_time"],
                }
                save_json(metadata, chain_dir / "metadata.json", indent=2)

            # Save plots
            if result.plots:
                plots_dir = output_dir / "plots"
                plots_dir.mkdir(exist_ok=True)

                from core.io_utils import save_plot

                for plot_name, fig in result.plots.items():
                    if fig is not None:
                        save_plot(plots_dir / f"{plot_name}.png", dpi=300, close_after=False)
                        save_plot(plots_dir / f"{plot_name}.pdf", close_after=True)
                        logger.info(f"  Saved plot: {plot_name}")

            logger.info(f"Results saved to {output_dir}")

        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
