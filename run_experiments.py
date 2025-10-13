#!/usr/bin/env python
"""
Remote Workstation Experiment Runner
Runs the complete experimental framework optimized for university GPU resources.

This modular version imports experiments from separate modules for better organization.
"""

from experiments.sgfa_configuration_comparison import run_sgfa_configuration_comparison
from experiments.sensitivity_analysis import run_sensitivity_analysis
from experiments.model_comparison import run_model_comparison
from experiments.data_validation import run_data_validation
from experiments.clinical_validation import run_clinical_validation
from experiments.robustness_testing import run_robustness_testing
from core.config_utils import (
    check_configuration_warnings,
    ensure_directories,
    get_output_dir,
    validate_configuration,
)
import argparse
import logging
import sys
from datetime import datetime

import yaml

# Add project root to path
sys.path.insert(0, ".")


# Custom FileHandler that handles stale file handles on network filesystems
class ResilientFileHandler(logging.FileHandler):
    """FileHandler that silently ignores OSError on flush (e.g., stale NFS handles)."""

    def flush(self):
        try:
            super().flush()
        except OSError:
            # Silently ignore stale file handle errors on network filesystems
            pass


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        ResilientFileHandler("logs/remote_workstation_experiments.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

# Import utilities and experiment functions


def load_config(config_path="config.yaml"):
    """Load remote workstation configuration."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f" Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f" Failed to load config: {e}")
        sys.exit(1)


def setup_environment(config):
    """Setup experimental environment."""
    # Create output directories using safe access
    ensure_directories(config)

    # Configure JAX based on config settings
    try:
        import jax
        import os

        # Check if GPU usage is disabled in config
        from core.config_utils import ConfigHelper
        config_dict = ConfigHelper.to_dict(config)
        system_config = config_dict.get("system", {})
        use_gpu = system_config.get("use_gpu", True)

        if not use_gpu:
            # Force JAX to use CPU only
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            jax.config.update("jax_platform_name", "cpu")
            logger.info(" Forcing CPU-only mode as configured")

        devices = jax.devices()
        logger.info(f" Available devices: {devices}")

        # Check for both 'gpu' and 'cuda' device types
        gpu_devices = [d for d in devices if d.platform in ["gpu", "cuda"]]
        if len(gpu_devices) == 0:
            logger.warning("  No GPU devices found - will use CPU (slower)")
        else:
            logger.info(f" GPU devices available for acceleration: {gpu_devices}")

    except Exception as e:
        logger.error(f" JAX setup issue: {e}")


def main():
    """Main experimental pipeline."""
    parser = argparse.ArgumentParser(
        description="Run Remote Workstation experimental framework"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=[
            "data_validation",
            "sgfa_configuration_comparison",
            "model_comparison",
            "sensitivity_analysis",
            "clinical_validation",
            "robustness_testing",
            "factor_stability",
            "neuroimaging_hyperopt",
            "neuroimaging_cv_benchmarks",
            "all",
            "all_extended",
        ],
        default=["all"],
        help="Experiments to run (use 'all' for core pipeline, 'all_extended' for full pipeline with comparison studies)",
    )
    parser.add_argument("--data-dir", help="Override data directory")
    parser.add_argument(
        "--unified-results",
        action="store_true",
        default=True,
        help="Save all results in a single timestamped folder (default: True)",
    )
    parser.add_argument(
        "--no-shared-data",
        action="store_true",
        default=False,
        help="Disable shared data pipeline (default: shared data enabled)",
    )
    parser.add_argument(
        "--independent-mode",
        action="store_true",
        default=False,
        help="Force independent data loading for troubleshooting (overrides shared data mode)",
    )
    parser.add_argument(
        "--select-rois",
        nargs="+",
        help="Select specific ROIs to load (e.g., --select-rois volume_sn_voxels.tsv)",
    )
    parser.add_argument(
        "--regress-confounds",
        nargs="+",
        help="Regress out confound variables from all views (e.g., --regress-confounds age sex tiv)",
    )
    parser.add_argument(
        "--drop-confounds-from-clinical",
        action="store_true",
        default=True,
        help="Drop confounds from clinical view instead of residualizing (default: True)",
    )
    parser.add_argument(
        "--residualize-confounds-in-clinical",
        dest="drop_confounds_from_clinical",
        action="store_false",
        help="Residualize confounds in clinical view instead of dropping them",
    )
    parser.add_argument(
        "--test-k",
        nargs="+",
        type=int,
        help="Specify K values to test in parameter comparison (e.g., --test-k 2 3)",
    )

    args = parser.parse_args()

    # Load and validate configuration
    config = load_config(args.config)

    logger.info("🔍 Validating configuration...")
    try:
        config = validate_configuration(config)
        logger.info("✅ Configuration validation passed")

        # Check for warnings
        warnings = check_configuration_warnings(config)
        for warning in warnings:
            logger.warning(f"⚠️  {warning}")

    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        sys.exit(1)

    # Override data directory if provided
    if args.data_dir:
        if "data" not in config:
            config["data"] = {}
        config["data"]["data_dir"] = args.data_dir
        logger.info(f"Using data directory: {args.data_dir}")

    # Configure ROI selection if provided
    if args.select_rois:
        if "preprocessing" not in config:
            config["preprocessing"] = {}
        config["preprocessing"]["select_rois"] = args.select_rois
        logger.info(f"Selecting ROIs: {args.select_rois}")

    # Configure confound regression if provided
    if args.regress_confounds:
        if "preprocessing" not in config:
            config["preprocessing"] = {}
        config["preprocessing"]["regress_confounds"] = args.regress_confounds
        config["preprocessing"]["drop_confounds_from_clinical"] = args.drop_confounds_from_clinical

        action = "dropping" if args.drop_confounds_from_clinical else "residualizing"
        logger.info(f"Regressing confounds: {args.regress_confounds} ({action} in clinical view)")

    # Configure K values for parameter comparison if provided
    # This overrides ALL n_factors settings across all experiments
    if args.test_k:
        logger.info(f"⚙️  Overriding all n_factors configurations with --test-k: {args.test_k}")

        # Override sgfa_configuration_comparison
        if "sgfa_configuration_comparison" not in config:
            config["sgfa_configuration_comparison"] = {}
        if "parameter_ranges" not in config["sgfa_configuration_comparison"]:
            config["sgfa_configuration_comparison"]["parameter_ranges"] = {}
        config["sgfa_configuration_comparison"]["parameter_ranges"]["n_factors"] = args.test_k

        # Override model_comparison
        if "model_comparison" in config and "models" in config["model_comparison"]:
            for model in config["model_comparison"]["models"]:
                if "n_factors" in model:
                    model["n_factors"] = args.test_k

        # Override sensitivity_analysis
        if "sensitivity_analysis" not in config:
            config["sensitivity_analysis"] = {}
        if "parameter_ranges" not in config["sensitivity_analysis"]:
            config["sensitivity_analysis"]["parameter_ranges"] = {}
        config["sensitivity_analysis"]["parameter_ranges"]["n_factors"] = args.test_k

        logger.info(f"✓ All n_factors overridden to: {args.test_k}")

    # Setup unified results directory if requested
    if args.unified_results:
        # Create single timestamped directory for all experiments
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = get_output_dir(config)

        # Build configuration suffix for directory name
        config_suffix = ""

        # Add ROI selection info
        if args.select_rois:
            # Extract ROI names (remove .tsv extension and volume_ prefix if present)
            roi_names = []
            for roi in args.select_rois:
                roi_name = roi.replace('.tsv', '').replace('volume_', '').replace('_voxels', '')
                roi_names.append(roi_name)
            config_suffix += f"_rois-{'+'.join(roi_names)}"

        # Add confound regression info
        if args.regress_confounds:
            config_suffix += f"_conf-{'+'.join(args.regress_confounds)}"
            if not args.drop_confounds_from_clinical:
                config_suffix += "_residualized"

        # Add K values info
        if args.test_k:
            config_suffix += f"_K-{'+'.join(map(str, args.test_k))}"

        # Create directory name based on what's actually running
        if len(args.experiments) == 1:
            unified_dir = output_dir / f"{args.experiments[0]}{config_suffix}_run_{run_timestamp}"
        elif len(args.experiments) < 5:
            unified_dir = output_dir / f"{'_'.join(args.experiments)}{config_suffix}_run_{run_timestamp}"
        else:
            unified_dir = output_dir / f"complete_run{config_suffix}_{run_timestamp}"
        unified_dir.mkdir(parents=True, exist_ok=True)

        # Update config to use unified directory
        if "experiments" not in config:
            config["experiments"] = {}
        config["experiments"]["base_output_dir"] = str(unified_dir)

        logger.info(f"🗂️  Using unified results directory: {unified_dir}")
        logger.info(f"   All experiments will save to: {unified_dir.name}")

        # Create organized subdirectories only for experiments that will actually run
        experiment_dir_mapping = {
            "data_validation": "01_data_validation",
            "sgfa_configuration_comparison": "02_sgfa_configuration_comparison",
            "model_comparison": "03_model_comparison",
            "sensitivity_analysis": "04_sensitivity_analysis",
            "clinical_validation": "05_clinical_validation",
            "robustness_testing": "06_robustness_testing",
            "factor_stability": "07_factor_stability"
        }

        # Only create directories for experiments that are actually being run
        for experiment in args.experiments:
            if experiment in experiment_dir_mapping:
                (unified_dir / experiment_dir_mapping[experiment]).mkdir(exist_ok=True)
                logger.info(f"Created directory for: {experiment}")

        # Always create common directories if any experiment is running
        if args.experiments:
            (unified_dir / "plots").mkdir(exist_ok=True)
            (unified_dir / "brain_maps").mkdir(exist_ok=True)
            (unified_dir / "summaries").mkdir(exist_ok=True)

    # Setup environment
    setup_environment(config)

    # Initialize SGFA results cache for sharing across experiments
    from core.results_cache import SGFAResultsCache
    output_dir = get_output_dir(config)
    cache_dir = output_dir / ".sgfa_cache"
    results_cache = SGFAResultsCache(cache_dir=cache_dir)

    # Add cache to config so experiments can access it
    if "experiments" not in config:
        config["experiments"] = {}
    config["experiments"]["results_cache"] = results_cache

    # Track results
    results = {}
    start_time = datetime.now()

    logger.info(f" Starting Remote Workstation Experimental Framework at {start_time}")
    logger.info(f" Running experiments: {args.experiments}")

    # Determine which experiments to run
    experiments_to_run = args.experiments
    if "all" in experiments_to_run:
        # Default: Core analysis pipeline (skip comparison/sensitivity studies)
        # Order: validate data → robustness testing → stability → clinical validation
        experiments_to_run = [
            "data_validation",
            "robustness_testing",
            "factor_stability",
            "clinical_validation",
        ]
        logger.info("ℹ️  Running core analysis pipeline (comparison studies skipped by default)")
        logger.info("   To include comparison studies, use: --experiments all_extended")

    if "all_extended" in experiments_to_run:
        # Extended: Include all comparison and sensitivity studies
        # Order: validate → compare configs → compare models → sensitivity → reproduce → stability → clinical
        experiments_to_run = [
            "data_validation",
            "sgfa_configuration_comparison",
            "model_comparison",
            "sensitivity_analysis",
            "robustness_testing",
            "factor_stability",
            "clinical_validation",
        ]
        logger.info("ℹ️  Running extended pipeline with all comparison and sensitivity studies")

    # Determine execution mode
    use_shared_data = not args.no_shared_data and not args.independent_mode
    if args.independent_mode:
        logger.info(
            "🔧 Using INDEPENDENT MODE - each experiment loads its own data (for troubleshooting)"
        )
    elif use_shared_data:
        logger.info("🔗 Using SHARED DATA MODE - efficient pipeline with data reuse")
    else:
        logger.info("🔧 Using INDEPENDENT MODE - shared data disabled")

    # Initialize pipeline context for data sharing
    pipeline_context = {
        "X_list": None,
        "preprocessing_info": None,
        "data_strategy": None,
        "shared_mode": use_shared_data,
        "memory_usage_mb": 0,
        "optimal_sgfa_params": None,  # Store optimal K, percW from parameter comparison
        "sgfa_performance_metrics": None,  # Store performance info for model comparison
    }

    logger.info(f"🔄 Pipeline context initialized (shared_mode: {use_shared_data})")

    # Run experiments sequentially, passing context through config
    if "data_validation" in experiments_to_run:
        logger.info("🔍 1/6 Starting Data Validation Experiment...")
        results["data_validation"] = run_data_validation(config)

        # Update pipeline context with results from data validation
        if results["data_validation"] and hasattr(
            results["data_validation"], "model_results"
        ):
            model_results = results["data_validation"].model_results
            if "preprocessed_data" in model_results:
                pipeline_context["X_list"] = model_results["preprocessed_data"].get(
                    "X_list"
                )
                pipeline_context["preprocessing_info"] = model_results[
                    "preprocessed_data"
                ].get("preprocessing_info")
                pipeline_context["data_strategy"] = model_results.get(
                    "data_strategy", "unknown"
                )

        # Log context update
        if pipeline_context["X_list"] is not None:
            logger.info(
                f"📊 Data validation loaded data: {len(pipeline_context['X_list'])} views"
            )
            logger.info(
                f"   Strategy: {pipeline_context.get('data_strategy', 'unknown')}"
            )

    if "robustness_testing" in experiments_to_run:
        logger.info("🔁 2/4 Starting Robustness Testing...")
        exp_config = config.copy()
        if pipeline_context["X_list"] is not None and use_shared_data:
            logger.info("   → Using shared data from previous experiments")

            # Extract view_names from preprocessing_info
            preprocessing_info = pipeline_context.get("preprocessing_info", {})
            view_names = preprocessing_info.get("data_summary", {}).get(
                "view_names",
                [f"view_{i}" for i in range(len(pipeline_context["X_list"]))]
            )
            logger.info(f"   Views: {view_names}")

            exp_config["_shared_data"] = {
                "X_list": pipeline_context["X_list"],
                "preprocessing_info": preprocessing_info,
                "view_names": view_names,  # Explicitly pass view_names
                "mode": "shared",
            }

            # Pass optimal SGFA parameters if available
            if pipeline_context["optimal_sgfa_params"] is not None:
                exp_config["_optimal_sgfa_params"] = pipeline_context[
                    "optimal_sgfa_params"
                ]
                logger.info(
                    f" → Using optimal SGFA params: { pipeline_context['optimal_sgfa_params']['variant_name']}"
                )

        results["robustness_testing"] = run_robustness_testing(exp_config)

    if "factor_stability" in experiments_to_run:
        logger.info("🔬 3/4 Starting Factor Stability Analysis...")
        exp_config = config.copy()
        if pipeline_context["X_list"] is not None and use_shared_data:
            logger.info("   → Using shared data from previous experiments")
            exp_config["_shared_data"] = {
                "X_list": pipeline_context["X_list"],
                "preprocessing_info": pipeline_context["preprocessing_info"],
                "mode": "shared",
            }

            # Pass optimal SGFA parameters if available
            if pipeline_context["optimal_sgfa_params"] is not None:
                exp_config["_optimal_sgfa_params"] = pipeline_context[
                    "optimal_sgfa_params"
                ]
                logger.info(
                    f"   → Using optimal SGFA params: {pipeline_context['optimal_sgfa_params']['variant_name']}"
                )

        # Import and run factor stability analysis
        from experiments.robustness_testing import ReproducibilityExperiments
        from experiments.framework import ExperimentConfig
        from core.config_utils import get_data_dir

        # Create experiment configuration
        fs_config = exp_config.get("factor_stability", {})
        experiment_config = ExperimentConfig(
            experiment_name="factor_stability_analysis",
            description="Factor stability analysis with fixed parameters (Ferreira et al. 2024)",
            dataset="qmap_pd",
            data_dir=get_data_dir(exp_config),
            K_values=[fs_config.get("K", 20)],
            num_samples=fs_config.get("num_samples", 5000),
            num_warmup=fs_config.get("num_warmup", 1000),
            num_chains=fs_config.get("num_chains", 4),
            cosine_threshold=fs_config.get("cosine_threshold", 0.8),
            min_match_rate=fs_config.get("min_match_rate", 0.5),
            sparsity_threshold=fs_config.get("sparsity_threshold", 0.01),
            min_nonzero_pct=fs_config.get("min_nonzero_pct", 0.05),
        )

        # Initialize reproducibility experiments
        repro_exp = ReproducibilityExperiments(experiment_config, logger)

        # Load data if not already available
        if exp_config.get("_shared_data") and exp_config["_shared_data"].get("X_list"):
            X_list = exp_config["_shared_data"]["X_list"]
            preprocessing_info = exp_config["_shared_data"].get("preprocessing_info", {})
            logger.info(f"   Using shared data: {len(X_list)} views")
        else:
            logger.info("   Loading data for factor stability analysis...")
            from data.preprocessing_integration import apply_preprocessing_to_pipeline

            X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                config=exp_config,
                data_dir=get_data_dir(exp_config),
                auto_select_strategy=False,
                preferred_strategy=exp_config.get("preprocessing", {}).get("strategy", "standard"),
            )
            logger.info(f"   Data loaded: {len(X_list)} views")

        # Extract view_names from preprocessing_info
        view_names = preprocessing_info.get("data_summary", {}).get("view_names", [f"view_{i}" for i in range(len(X_list))])
        logger.info(f"   Views: {view_names}")

        # Prepare hyperparameters (read from config.yaml)
        K = fs_config.get("K", 20)
        hypers = {
            "Dm": [X.shape[1] for X in X_list],
            "K": K,
            "a_sigma": 1.0,
            "b_sigma": 1.0,
            "slab_df": fs_config.get("slab_df", 4),
            "slab_scale": fs_config.get("slab_scale", 2),
            "percW": fs_config.get("percW", 33),
        }

        # Prepare MCMC args (read from config.yaml)
        mcmc_args = {
            "K": K,
            "num_warmup": fs_config.get("num_warmup", 1000),
            "num_samples": fs_config.get("num_samples", 5000),
            "num_chains": 1,  # Sequential execution (run_factor_stability_analysis handles multiple chains)
            "target_accept_prob": fs_config.get("target_accept_prob", 0.8),
            "reghsZ": fs_config.get("reghsZ", True),
            "random_seed": 42,
        }

        logger.info(f"   Using parameters from config.yaml:")
        logger.info(f"   - K={K}, percW={hypers['percW']}, slab_df={hypers['slab_df']}, slab_scale={hypers['slab_scale']}")
        logger.info(f"   - num_samples={mcmc_args['num_samples']}, num_warmup={mcmc_args['num_warmup']}")
        logger.info(f"   - cosine_threshold={experiment_config.cosine_threshold}, min_match_rate={experiment_config.min_match_rate}")

        # Run factor stability analysis
        logger.info(f"   Running {experiment_config.num_chains} chains with K={K} factors...")
        result = repro_exp.run_factor_stability_analysis(
            X_list=X_list,
            hypers=hypers,
            args=mcmc_args,
            n_chains=experiment_config.num_chains,
            cosine_threshold=experiment_config.cosine_threshold,
            min_match_rate=experiment_config.min_match_rate,
            view_names=view_names,  # Pass view names for plotting
        )

        results["factor_stability"] = result

        # Save W and Z matrices for each chain + stability analysis results
        if result and hasattr(result, "model_results"):
            logger.info("   💾 Saving factor loadings and scores...")

            # Get output directory for factor stability
            if unified_dir:
                fs_output_dir = unified_dir / experiment_dir_mapping.get("factor_stability", "07_factor_stability")
            else:
                fs_output_dir = get_output_dir(exp_config) / "factor_stability"
            fs_output_dir.mkdir(parents=True, exist_ok=True)

            # Import saving utilities
            from analysis.factor_stability import save_stability_results
            from core.io_utils import save_csv, save_json
            import pandas as pd

            # Extract results
            chain_results_data = result.model_results.get("chain_results", [])
            stability_results_data = result.model_results.get("stability_results", {})
            effective_factors_data = result.model_results.get("effective_factors", [])

            # Save chain results (W and Z for each chain)
            chains_dir = fs_output_dir / "chains"
            chains_dir.mkdir(exist_ok=True)

            for chain_data in chain_results_data:
                chain_id = chain_data.get("chain_id", 0)
                chain_dir = chains_dir / f"chain_{chain_id}"
                chain_dir.mkdir(exist_ok=True)

                # Save W (factor loadings)
                W = chain_data.get("W")
                if W is not None:
                    if isinstance(W, list):
                        # Multi-view: save each view
                        for view_idx, W_view in enumerate(W):
                            W_df = pd.DataFrame(
                                W_view,
                                columns=[f"Factor_{k}" for k in range(W_view.shape[1])],
                            )
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
                Z = chain_data.get("Z")
                if Z is not None:
                    Z_df = pd.DataFrame(
                        Z,
                        columns=[f"Factor_{k}" for k in range(Z.shape[1])],
                    )
                    Z_df.index = [f"Subject_{j}" for j in range(Z.shape[0])]
                    Z_df.index.name = "Subject"
                    save_csv(Z_df, chain_dir / "Z.csv", index=True)

                # Save metadata
                metadata = {
                    "chain_id": chain_data.get("chain_id", chain_id),
                    "seed": chain_data.get("seed", "unknown"),
                    "log_likelihood": float(chain_data.get("log_likelihood", 0)),
                    "convergence": chain_data.get("convergence", False),
                    "execution_time": chain_data.get("execution_time", 0),
                }
                save_json(metadata, chain_dir / "metadata.json", indent=2)

            # Save stability analysis results
            if stability_results_data and effective_factors_data:
                stability_dir = fs_output_dir / "stability_analysis"
                save_stability_results(
                    stability_results_data,
                    effective_factors_data[0] if effective_factors_data else {},
                    str(stability_dir),
                )

            # Save plots
            if hasattr(result, "plots") and result.plots:
                plots_dir = fs_output_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                from core.io_utils import save_plot
                for plot_name, fig in result.plots.items():
                    if fig is not None:
                        save_plot(plots_dir / f"{plot_name}.png", dpi=300, close_after=False)
                        save_plot(plots_dir / f"{plot_name}.pdf", close_after=True)

            logger.info(f"   ✓ Results saved to: {fs_output_dir}")

            # Log summary
            stability = result.model_results.get("stability_results", {})
            logger.info(f"   ✓ Found {stability.get('n_stable_factors', 0)}/{stability.get('total_factors', 0)} stable factors")
            logger.info(f"   ✓ Stability rate: {stability.get('stability_rate', 0):.1%}")
            logger.info(f"   ✓ Factor loadings (W) and scores (Z) saved for all {len(chain_results_data)} chains")

    if "clinical_validation" in experiments_to_run:
        logger.info("🏥 4/4 Starting Clinical Validation with Neuroimaging CV...")
        exp_config = config.copy()
        if pipeline_context["X_list"] is not None and use_shared_data:
            logger.info("   → Using shared data from previous experiments")

            # Extract view_names from preprocessing_info
            preprocessing_info = pipeline_context.get("preprocessing_info", {})
            view_names = preprocessing_info.get("data_summary", {}).get(
                "view_names",
                [f"view_{i}" for i in range(len(pipeline_context["X_list"]))]
            )
            logger.info(f"   Views: {view_names}")

            exp_config["_shared_data"] = {
                "X_list": pipeline_context["X_list"],
                "preprocessing_info": preprocessing_info,
                "view_names": view_names,  # Explicitly pass view_names
                "mode": "shared",
            }

            # Pass optimal SGFA parameters if available
            if pipeline_context["optimal_sgfa_params"] is not None:
                exp_config["_optimal_sgfa_params"] = pipeline_context[
                    "optimal_sgfa_params"
                ]
                logger.info(
                    f" → Using optimal SGFA params: { pipeline_context['optimal_sgfa_params']['variant_name']}"
                )

        results["clinical_validation"] = run_clinical_validation(exp_config)

    if "sgfa_configuration_comparison" in experiments_to_run:
        logger.info("🔬 2/6 Starting SGFA Hyperparameter Tuning Experiment...")
        exp_config = config.copy()
        if pipeline_context["X_list"] is not None and use_shared_data:
            logger.info("   → Using shared data from data_validation")
            exp_config["_shared_data"] = {
                "X_list": pipeline_context["X_list"],
                "preprocessing_info": pipeline_context["preprocessing_info"],
                "mode": "shared",
            }
        sgfa_result = run_sgfa_configuration_comparison(exp_config)
        results["sgfa_configuration_comparison"] = sgfa_result

        # Extract optimal parameters for downstream experiments
        if sgfa_result and use_shared_data:
            try:
                # Extract best performing variant info
                # ExperimentResult is a dataclass, access attributes not dict keys
                if hasattr(sgfa_result, "model_results") and sgfa_result.model_results:
                    model_results = sgfa_result.model_results
                    if "sgfa_variants" in model_results:
                        # Find best variant by execution time and convergence
                        best_variant = None
                        best_score = float("inf")

                        for variant_name, variant_data in model_results[
                            "sgfa_variants"
                        ].items():
                            if variant_data.get("convergence", False):
                                exec_time = variant_data.get(
                                    "execution_time", float("inf")
                                )
                                if exec_time < best_score:
                                    best_score = exec_time
                                    best_variant = variant_name

                        if best_variant:
                            # Parse K, percW, and group_lambda from variant name (e.g., "K5_percW25_grp0.0")
                            import re

                            match = re.match(r"K(\d+)_percW([\d.]+)(?:_grp([\d.]+))?", best_variant)
                            if match:
                                optimal_K = int(match.group(1))
                                optimal_percW = float(match.group(2))
                                optimal_grp_lambda = float(match.group(3)) if match.group(3) else 0.0

                                pipeline_context["optimal_sgfa_params"] = {
                                    "K": optimal_K,
                                    "percW": optimal_percW,
                                    "grp_lambda": optimal_grp_lambda,
                                    "variant_name": best_variant,
                                    "execution_time": best_score,
                                }

                                logger.info(
                                    f"🎯 Identified optimal SGFA parameters: {best_variant} (K={optimal_K}, percW={optimal_percW}, grp_λ={optimal_grp_lambda}, {best_score:.1f}s)"
                                )

            except Exception as e:
                logger.warning(f"Could not extract optimal parameters: {e}")
                pipeline_context["optimal_sgfa_params"] = None

    if "model_comparison" in experiments_to_run:
        logger.info("🧠 3/6 Starting Model Architecture Comparison Experiment...")
        exp_config = config.copy()

        # Debug: Log data sharing status
        logger.info(f"   DEBUG: X_list available: {pipeline_context['X_list'] is not None}")
        logger.info(f"   DEBUG: use_shared_data: {use_shared_data}")
        logger.info(f"   DEBUG: optimal_sgfa_params available: {pipeline_context['optimal_sgfa_params'] is not None}")

        if pipeline_context["X_list"] is not None and use_shared_data:
            logger.info("   → Using shared data from previous experiments")
            exp_config["_shared_data"] = {
                "X_list": pipeline_context["X_list"],
                "preprocessing_info": pipeline_context["preprocessing_info"],
                "mode": "shared",
            }

            # Pass optimal SGFA parameters if available
            if pipeline_context["optimal_sgfa_params"] is not None:
                exp_config["_optimal_sgfa_params"] = pipeline_context[
                    "optimal_sgfa_params"
                ]
                logger.info(
                    f" → Using optimal SGFA params: { pipeline_context['optimal_sgfa_params']['variant_name']}"
                )

        results["model_comparison"] = run_model_comparison(exp_config)

    if "sensitivity_analysis" in experiments_to_run:
        logger.info("📊 4/6 Starting Sensitivity Analysis Experiment...")
        exp_config = config.copy()
        if pipeline_context["X_list"] is not None and use_shared_data:
            logger.info("   → Using shared data from previous experiments")
            exp_config["_shared_data"] = {
                "X_list": pipeline_context["X_list"],
                "preprocessing_info": pipeline_context["preprocessing_info"],
                "mode": "shared",
            }

            # Pass optimal SGFA parameters if available
            if pipeline_context["optimal_sgfa_params"] is not None:
                exp_config["_optimal_sgfa_params"] = pipeline_context[
                    "optimal_sgfa_params"
                ]
                logger.info(
                    f" → Using optimal SGFA params: { pipeline_context['optimal_sgfa_params']['variant_name']}"
                )

        results["sensitivity_analysis"] = run_sensitivity_analysis(exp_config)

    if "neuroimaging_hyperopt" in experiments_to_run:
        logger.info("🔬 7/8 Starting Neuroimaging Hyperparameter Optimization...")
        exp_config = config.copy()
        if pipeline_context["X_list"] is not None and use_shared_data:
            logger.info("   → Using shared data from previous experiments")
            exp_config["_shared_data"] = {
                "X_list": pipeline_context["X_list"],
                "preprocessing_info": pipeline_context["preprocessing_info"],
                "mode": "shared",
            }

        # Run neuroimaging configuration optimization from sgfa_configuration_comparison
        from experiments.sgfa_configuration_comparison import SGFAConfigurationComparison
        from experiments.framework import ExperimentConfig

        experiment_config = ExperimentConfig.from_dict(exp_config)
        sgfa_exp = SGFAConfigurationComparison(experiment_config)

        # Generate synthetic data if shared data not available
        if pipeline_context["X_list"] is None:
            logger.info("   → No shared data available, will use synthetic data")

        results["neuroimaging_hyperopt"] = sgfa_exp.run_neuroimaging_hyperparameter_optimization(
            X_list=pipeline_context.get("X_list"),
            hypers=exp_config.get("hypers", {}),
            args=exp_config.get("args", {})
        )

    if "neuroimaging_cv_benchmarks" in experiments_to_run:
        logger.info("📊 8/8 Starting Neuroimaging CV Benchmarks...")
        exp_config = config.copy()
        if pipeline_context["X_list"] is not None and use_shared_data:
            logger.info("   → Using shared data from previous experiments")
            exp_config["_shared_data"] = {
                "X_list": pipeline_context["X_list"],
                "preprocessing_info": pipeline_context["preprocessing_info"],
                "mode": "shared",
            }

        # Run clinical-aware CV benchmarks from clinical_validation (functionality moved)
        from experiments.clinical_validation import ClinicalValidationExperiments
        from experiments.framework import ExperimentConfig

        experiment_config = ExperimentConfig.from_dict(exp_config)
        clinical_exp = ClinicalValidationExperiments(experiment_config)

        # Use integrated SGFA + clinical validation instead of separate CV benchmarks
        results["neuroimaging_cv_benchmarks"] = clinical_exp.run_sgfa_clinical_validation(
            X_base=pipeline_context.get("X_list"),
            hypers=exp_config.get("hypers", {}),
            args=exp_config.get("args", {})
        )

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    # Clear SGFA results cache
    if results_cache:
        cache_stats = results_cache.get_stats()
        logger.info(f"📊 Cache statistics: {cache_stats['memory_entries']} memory entries, {cache_stats['disk_entries']} disk entries")
        results_cache.clear()
        logger.info("🗑️  SGFA results cache cleared")

    logger.info(f" All experiments completed!")
    logger.info(f"  Total duration: {duration}")
    logger.info(f" Results saved to: {get_output_dir(config)}")

    # Create comprehensive summary
    if args.unified_results:
        logger.info(f"📋 Creating comprehensive experiment summary...")

        # Collect detailed results information
        summary = {
            "experiment_run_info": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "duration_formatted": str(duration),
                "unified_results": True,
                "config_used": args.config,
            },
            "experiments_executed": {},
            "results_summary": {
                "total_experiments": len(experiments_to_run),
                "successful_experiments": len(
                    [r for r in results.values() if r is not None]
                ),
                "failed_experiments": len([r for r in results.values() if r is None]),
            },
        }

        # Add experiment-specific summaries
        for exp_name, result in results.items():
            if result is not None:
                exp_summary = {
                    "status": getattr(result, "status", "completed"),
                    "experiment_id": getattr(result, "experiment_id", "N/A"),
                    "duration": getattr(result, "get_duration", lambda: 0)(),
                }

                # Add specific details based on experiment type
                if exp_name == "sgfa_configuration_comparison" and hasattr(
                    result, "model_results"
                ):
                    model_results = result.model_results
                    if "sgfa_variants" in model_results:
                        exp_summary["sgfa_variants"] = list(
                            model_results["sgfa_variants"].keys()
                        )
                        exp_summary["successful_variants"] = len(
                            [
                                v
                                for v in model_results["sgfa_variants"].values()
                                if v.get("status") == "completed"
                            ]
                        )
                    if "plots" in model_results:
                        exp_summary["plots_generated"] = model_results["plots"].get(
                            "plot_count", 0
                        )
                        exp_summary["brain_maps_available"] = "brain_maps" in str(
                            model_results["plots"].get("generated_plots", [])
                        )

                elif exp_name == "model_comparison" and hasattr(
                    result, "model_results"
                ):
                    model_results = result.model_results
                    if "sparseGFA" in model_results:
                        exp_summary["implemented_models"] = [
                            k
                            for k, v in model_results.items()
                            if not v.get("skipped", False)
                        ]
                        exp_summary["future_work_models"] = [
                            k
                            for k, v in model_results.items()
                            if v.get("skipped", False)
                        ]
                    if "traditional_methods" in model_results:
                        exp_summary["traditional_methods"] = list(
                            model_results["traditional_methods"].keys()
                        )

                summary["experiments_executed"][exp_name] = exp_summary
            else:
                summary["experiments_executed"][exp_name] = {"status": "failed"}

        # Save main summary
        summary_path = (
            get_output_dir(config) / "summaries" / "complete_experiment_summary.yaml"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

        # Create a simple text summary for quick reading
        text_summary_path = get_output_dir(config) / "README.md"
        with open(text_summary_path, "w") as f:
            f.write(f"# SGFA Experiment Run Results\\n\\n")
            f.write(f"**Run Date:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**Duration:** {duration}\\n")
            f.write(f"**Experiments:** {', '.join(experiments_to_run)}\\n\\n")

            f.write(f"## Results Structure\\n\\n")
            f.write(f"```\\n")

            # Only document directories for experiments that were actually run
            experiment_descriptions = {
                "data_validation": "01_data_validation/     - Data quality and preprocessing analysis",
                "sgfa_configuration_comparison": "02_sgfa_configuration_comparison/   - SGFA hyperparameter optimization",
                "model_comparison": "03_model_comparison/   - Model architecture comparison",
                "sensitivity_analysis": "04_sensitivity_analysis/ - Parameter sensitivity studies",
                "clinical_validation": "05_clinical_validation/ - Clinical validation studies",
                "robustness_testing": "06_robustness_testing/     - Robustness and quality control testing"
            }

            for experiment in experiments_to_run:
                if experiment in experiment_descriptions:
                    f.write(f"{experiment_descriptions[experiment]}\\n")

            # Always document common directories if any experiments ran
            if experiments_to_run:
                f.write(f"plots/                  - All visualization outputs\\n")
                f.write(f"brain_maps/            - Factor loadings mapped to brain space\\n")
                f.write(f"summaries/             - Detailed summaries and reports\\n")

            f.write(f"```\\n\\n")

            # Add experiment-specific details
            for exp_name, result in results.items():
                if result is not None:
                    f.write(f"### {exp_name.replace('_', ' ').title()}\\n")
                    f.write(f"- Status: Completed\\n")
                    if exp_name == "sgfa_configuration_comparison" and hasattr(
                        result, "model_results"
                    ):
                        model_results = result.model_results
                        if "sgfa_variants" in model_results:
                            f.write(
                                f"- SGFA Parameter Variants: {list(model_results['sgfa_variants'].keys())}\\n"
                            )
                        if "plots" in model_results:
                            f.write(
                                f"- Plots Generated: {model_results['plots'].get('plot_count', 0)}\\n"
                            )

                    elif exp_name == "model_comparison" and hasattr(
                        result, "model_results"
                    ):
                        model_results = result.model_results
                        implemented_models = [
                            k
                            for k, v in model_results.items()
                            if not v.get("skipped", False)
                        ]
                        future_models = [
                            k
                            for k, v in model_results.items()
                            if v.get("skipped", False)
                        ]
                        if implemented_models:
                            f.write(f"- Implemented Models: {implemented_models}\\n")
                        if future_models:
                            f.write(f"- Future Work Models: {future_models}\\n")
                    f.write(f"\\n")
                else:
                    f.write(f"### {exp_name.replace('_', ' ').title()}\\n")
                    f.write(f"- Status: Failed\\n\\n")

        logger.info(f"📋 Comprehensive summary saved to: {summary_path}")
        logger.info(f"📖 Quick reference saved to: {text_summary_path}")

    else:
        # Simple summary for non-unified results
        summary = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "experiments_run": experiments_to_run,
            "success_count": len([r for r in results.values() if r is not None]),
            "config_used": args.config,
        }

        summary_path = get_output_dir(config) / "experiment_summary.yaml"
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

        logger.info(f" Experiment summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
