#!/usr/bin/env python
"""
Remote Workstation Experiment Runner
Runs the complete experimental framework optimized for university GPU resources.

This modular version imports experiments from separate modules for better organization.
"""

from experiments.data_validation import run_data_validation
from experiments.robustness_testing import run_robustness_testing
from experiments.clinical_validation import run_clinical_validation
from core.config_utils import (
    check_configuration_warnings,
    ensure_directories,
    get_output_dir,
    validate_configuration,
)
import argparse
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

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


# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

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
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
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
            "robustness_testing",
            "factor_stability",
            "clinical_validation",
            "all",
        ],
        default=["all"],
        help="Experiments to run (default: 'all' runs the core analysis pipeline)",
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
        "--feature-selection",
        choices=["none", "variance", "statistical", "mutual_info", "combined"],
        default=None,
        help="Feature selection method (default: use config.yaml setting). Options: none (no selection), variance (remove low-variance features), statistical, mutual_info, combined",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=None,
        help="Variance threshold for feature selection (e.g., 0.02 to remove features with <2%% variance). Only used with --feature-selection variance",
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
        "--qc-outlier-threshold",
        type=float,
        default=None,
        help="MAD threshold for QC outlier detection (default: 3.0). Higher values are more permissive. Example: --qc-outlier-threshold 5.0 to keep more voxels",
    )

    # PCA and model configuration arguments
    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help="Number of latent factors (overrides config.yaml). Example: --K 8",
    )
    parser.add_argument(
        "--percW",
        type=float,
        default=None,
        help="Sparsity level - percentage of non-zero loadings (overrides config.yaml). Example: --percW 20 for 20%% non-zero (80%% sparse)",
    )
    parser.add_argument(
        "--enable-pca",
        action="store_true",
        default=False,
        help="Enable PCA dimensionality reduction",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=None,
        help="PCA variance threshold (e.g., 0.85 for 85%% variance). Example: --pca-variance 0.85",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="Fixed number of PCA components (alternative to --pca-variance). Example: --pca-components 120",
    )
    parser.add_argument(
        "--pca-strategy",
        choices=["aggressive", "balanced", "conservative"],
        default=None,
        help="Pre-configured PCA strategy: aggressive (80%% var), balanced (85%% var), conservative (90%% var)",
    )

    # MCMC configuration arguments
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=None,
        help="NUTS maximum tree depth (default: 13). Higher values allow longer trajectories but use more memory. Example: --max-tree-depth 15 for more thorough sampling",
    )
    parser.add_argument(
        "--target-accept-prob",
        type=float,
        default=None,
        help="MCMC target acceptance probability (default: 0.8). Higher values (0.9-0.99) for better sampling. Example: --target-accept-prob 0.95",
    )
    parser.add_argument(
        "--use-pca-initialization",
        action="store_true",
        default=False,
        help="Enable PCA-based initialization for MCMC chains (overrides config.yaml). Uses PCA to provide smart starting values for factor loadings and scores.",
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
        logger.info(f"Confound regression: {action} {', '.join(args.regress_confounds)}")

    # Configure feature selection if provided
    if args.feature_selection:
        if "preprocessing" not in config:
            config["preprocessing"] = {}
        config["preprocessing"]["feature_selection_method"] = args.feature_selection
        logger.info(f"Feature selection method: {args.feature_selection}")

        if args.variance_threshold is not None:
            config["preprocessing"]["variance_threshold"] = args.variance_threshold
            logger.info(f"Variance threshold: {args.variance_threshold}")

    # Configure QC outlier threshold if provided
    if args.qc_outlier_threshold is not None:
        if "preprocessing" not in config:
            config["preprocessing"] = {}
        config["preprocessing"]["qc_outlier_threshold"] = args.qc_outlier_threshold
        config["preprocessing"]["enable_spatial_processing"] = True  # Required for MAD filtering
        logger.info(f"QC outlier threshold (MAD): {args.qc_outlier_threshold}")
        logger.info(f"   Enabled spatial processing for MAD filtering")
    else:
        # If no MAD threshold specified, disable MAD filtering (set to None)
        # This allows PCA without MAD filtering
        if "preprocessing" not in config:
            config["preprocessing"] = {}
        if "qc_outlier_threshold" not in config["preprocessing"]:
            config["preprocessing"]["qc_outlier_threshold"] = None

    # Configure K (number of factors) if provided
    if args.K is not None:
        if "model" not in config:
            config["model"] = {}
        config["model"]["K"] = args.K
        logger.info(f"Override K (latent factors): {args.K}")

    # Configure percW (sparsity level) if provided
    if args.percW is not None:
        if "model" not in config:
            config["model"] = {}
        config["model"]["percW"] = args.percW
        logger.info(f"Override percW (sparsity): {args.percW}% non-zero ({100-args.percW:.0f}% sparse)")

    # Configure max_tree_depth if provided
    if args.max_tree_depth is not None:
        if "mcmc" not in config:
            config["mcmc"] = {}
        config["mcmc"]["max_tree_depth"] = args.max_tree_depth
        logger.info(f"Override max_tree_depth: {args.max_tree_depth}")

    # Configure target_accept_prob if provided
    if args.target_accept_prob is not None:
        if "mcmc" not in config:
            config["mcmc"] = {}
        config["mcmc"]["target_accept_prob"] = args.target_accept_prob
        # Also set it in factor_stability section for consistency
        if "factor_stability" in config:
            config["factor_stability"]["target_accept_prob"] = args.target_accept_prob
        logger.info(f"Override target_accept_prob: {args.target_accept_prob}")

    # Configure PCA initialization if provided
    if args.use_pca_initialization:
        if "model" not in config:
            config["model"] = {}
        config["model"]["use_pca_initialization"] = True
        logger.info(f"Override use_pca_initialization: True (PCA-based MCMC initialization enabled)")

    # Configure PCA if provided
    if args.enable_pca or args.pca_strategy or args.pca_variance or args.pca_components:
        if "preprocessing" not in config:
            config["preprocessing"] = {}

        # Enable spatial processing (required for PCA on imaging data)
        config["preprocessing"]["enable_spatial_processing"] = True

        # Handle PCA strategy shorthand
        if args.pca_strategy:
            strategy_map = {
                "aggressive": 0.80,
                "balanced": 0.85,
                "conservative": 0.90,
            }
            config["preprocessing"]["enable_pca"] = True
            config["preprocessing"]["pca_n_components"] = None
            config["preprocessing"]["pca_variance_threshold"] = strategy_map[args.pca_strategy]
            config["preprocessing"]["pca_whiten"] = False
            logger.info(f"PCA strategy: {args.pca_strategy} ({strategy_map[args.pca_strategy]*100:.0f}% variance)")
            logger.info(f"   Enabled spatial processing for PCA")

        # Handle explicit PCA configuration
        elif args.enable_pca:
            config["preprocessing"]["enable_pca"] = True

            if args.pca_components:
                config["preprocessing"]["pca_n_components"] = args.pca_components
                config["preprocessing"]["pca_variance_threshold"] = None
                logger.info(f"PCA enabled: {args.pca_components} components (fixed)")
            elif args.pca_variance:
                config["preprocessing"]["pca_n_components"] = None
                config["preprocessing"]["pca_variance_threshold"] = args.pca_variance
                logger.info(f"PCA enabled: {args.pca_variance*100:.0f}% variance threshold")
            else:
                # Default to balanced strategy
                config["preprocessing"]["pca_n_components"] = None
                config["preprocessing"]["pca_variance_threshold"] = 0.85
                logger.info(f"PCA enabled: 85% variance (default)")

            config["preprocessing"]["pca_whiten"] = False
            logger.info(f"   Enabled spatial processing for PCA")

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

        # Add K info if overridden
        if args.K:
            config_suffix += f"_K{args.K}"

        # Add percW info if overridden
        if args.percW:
            config_suffix += f"_percW{args.percW:.0f}"

        # Add PCA info
        if args.pca_strategy:
            config_suffix += f"_PCA-{args.pca_strategy}"
        elif args.enable_pca:
            if args.pca_components:
                config_suffix += f"_PCA-{args.pca_components}comp"
            elif args.pca_variance:
                config_suffix += f"_PCA-{int(args.pca_variance*100)}pct"
            else:
                config_suffix += "_PCA-85pct"

        # Add MAD threshold info
        if args.qc_outlier_threshold:
            config_suffix += f"_MAD{args.qc_outlier_threshold}"

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

        # Build semantic names for subdirectories
        # Helper function to create semantic experiment names
        def get_semantic_subdir_name(base_name: str, prefix: str) -> str:
            """Create semantic subdirectory name with experiment parameters."""
            parts = [prefix]

            # Add model type abbreviation if not default
            model_type = config.get("model", {}).get("model_type", "sparseGFA")
            if model_type == "sparse_gfa_fixed":
                parts.append("fixed")
            elif model_type == "neuroGFA":
                parts.append("neuro")
            elif model_type in ["standard_gfa", "GFA"]:
                parts.append("std")
            # sparseGFA is default, no abbreviation

            # Add key parameters
            model_config_params = config.get("model", {})
            if "K" in model_config_params:
                parts.append(f"K{model_config_params['K']}")
            if "percW" in model_config_params:
                parts.append(f"percW{model_config_params['percW']:.0f}")

            # Add slab parameters
            slab_df = model_config_params.get("slab_df")
            slab_scale = model_config_params.get("slab_scale")
            if slab_df and slab_scale:
                parts.append(f"slab{slab_df:.0f}_{slab_scale:.0f}")

            # Add MAD threshold
            mad_threshold = config.get("preprocessing", {}).get("qc_outlier_threshold")
            if mad_threshold:
                parts.append(f"MAD{mad_threshold:.1f}")

            # Add tree depth if non-default
            tree_depth = config.get("mcmc", {}).get("max_tree_depth") or \
                        config.get("factor_stability", {}).get("max_tree_depth")
            if tree_depth and tree_depth != 13:
                parts.append(f"tree{tree_depth}")

            return "_".join(parts)

        # Create organized subdirectories for pipeline experiments
        # Numbering reflects the pipeline execution order (data_validation → stability → clinical)
        # NOTE: robustness_testing was removed from the pipeline, so factor_stability is now 02
        experiment_dir_mapping = {
            "data_validation": get_semantic_subdir_name("data_validation", "01_data_validation"),
            "factor_stability": get_semantic_subdir_name("factor_stability", "02_factor_stability"),
            "clinical_validation": get_semantic_subdir_name("clinical_validation", "03_clinical_validation"),
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
        # Default: Core analysis pipeline
        # Order: validate data → factor stability
        # Note: robustness_testing available but skipped by default (only 150 iterations, not a real analysis)
        experiments_to_run = [
            "data_validation",
            "factor_stability",
        ]
        logger.info("ℹ️  Running core analysis pipeline: data_validation → factor_stability")

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
        logger.info("🔍 1/2 Starting Data Validation Experiment...")
        # Pass unified directory and semantic experiment directory name
        if 'unified_dir' in locals() and unified_dir:
            data_val_dir = experiment_dir_mapping.get("data_validation", "01_data_validation")
            results["data_validation"] = run_data_validation(config, unified_dir=unified_dir, experiment_dir=data_val_dir)
        else:
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

            # Extract view_names and feature_names from preprocessing_info
            preprocessing_info = pipeline_context.get("preprocessing_info", {})
            view_names = preprocessing_info.get("data_summary", {}).get(
                "view_names",
                [f"view_{i}" for i in range(len(pipeline_context["X_list"]))]
            )
            feature_names = preprocessing_info.get("data_summary", {}).get("original_data", {}).get("feature_names", {})
            logger.info(f"   Views: {view_names}")

            exp_config["_shared_data"] = {
                "X_list": pipeline_context["X_list"],
                "preprocessing_info": preprocessing_info,
                "view_names": view_names,  # Explicitly pass view_names
                "feature_names": feature_names,  # Explicitly pass feature_names
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
        logger.info("=" * 80)
        logger.info("🔬 3/4 STARTING FACTOR STABILITY ANALYSIS")
        logger.info("=" * 80)
        exp_config = config.copy()
        if pipeline_context["X_list"] is not None and use_shared_data:
            logger.info("   → Using shared data from previous experiments")
            logger.info(f"   → X_list: {len(pipeline_context['X_list'])} views")
            for i, X in enumerate(pipeline_context["X_list"]):
                logger.info(f"      View {i}: {X.shape}")
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
        else:
            logger.info("   → Will load fresh data")

        # Import and run factor stability analysis
        from experiments.robustness_testing import RobustnessExperiments
        from experiments.framework import ExperimentConfig
        from core.config_utils import get_data_dir

        # Create experiment configuration
        fs_config = exp_config.get("factor_stability", {})
        logger.info(f"   Factor stability config from config.yaml: K={fs_config.get('K')}, percW={fs_config.get('percW')}, num_chains={fs_config.get('num_chains')}")

        # Check for command-line K override and read hyperparameters
        model_config = exp_config.get("model", {})
        K_override = model_config.get("K", None)
        K_value = K_override if K_override is not None else fs_config.get("K", 20)
        if K_override is not None:
            logger.info(f"   Using command-line K override: {K_value} (config.yaml has {fs_config.get('K', 20)})")

        # Read other model hyperparameters for semantic naming
        percW_value = model_config.get("percW", fs_config.get("percW", 33))
        slab_df_value = model_config.get("slab_df", 4)
        slab_scale_value = model_config.get("slab_scale", 2)

        # Get QC outlier threshold (MAD) from preprocessing config for semantic naming
        preprocessing_config = exp_config.get("preprocessing", {})
        qc_outlier_threshold = preprocessing_config.get("qc_outlier_threshold", None)
        if qc_outlier_threshold:
            logger.info(f"   QC outlier threshold (MAD): {qc_outlier_threshold}")

        # Get model configuration
        model_config = exp_config.get("model", {})
        model_type = model_config.get("model_type", "sparseGFA")

        experiment_config = ExperimentConfig(
            experiment_name="factor_stability_analysis",
            description="Factor stability analysis with fixed parameters (Ferreira et al. 2024)",
            dataset="qmap_pd",
            data_dir=get_data_dir(exp_config),
            model_type=model_type,  # Use model_type from config
            K_values=[K_value],
            K=K_value,  # For semantic naming
            percW=percW_value,  # For semantic naming
            slab_df=slab_df_value,  # For semantic naming
            slab_scale=slab_scale_value,  # For semantic naming
            qc_outlier_threshold=qc_outlier_threshold,  # For semantic naming
            max_tree_depth=exp_config.get("mcmc", {}).get("max_tree_depth"),  # For semantic naming and MCMC config
            num_samples=fs_config.get("num_samples", 5000),
            num_warmup=fs_config.get("num_warmup", 1000),
            num_chains=fs_config.get("num_chains", 4),
            cosine_threshold=fs_config.get("cosine_threshold", 0.8),
            min_match_rate=fs_config.get("min_match_rate", 0.5),
            sparsity_threshold=fs_config.get("sparsity_threshold", 0.001),  # Adjusted for standardized data
            min_nonzero_pct=fs_config.get("min_nonzero_pct", 0.01),        # At least 1% of features
        )

        # Initialize robustness experiments
        repro_exp = RobustnessExperiments(experiment_config, logger)

        # Override base_output_dir with the run directory (unified_dir)
        # When running with --experiments factor_stability, we need to ensure
        # repro_exp uses the run directory, not the global results/ directory
        if 'unified_dir' in locals() and unified_dir:
            repro_exp.base_output_dir = unified_dir
            logger.info(f"📁 Set base_output_dir to run directory: {unified_dir}")

        # Create experiment-specific subdirectory inside the run directory
        # Use the same directory name from experiment_dir_mapping to avoid duplicate saving
        semantic_name = experiment_dir_mapping.get("factor_stability", "02_factor_stability")
        experiment_output_dir = repro_exp.base_output_dir / semantic_name
        experiment_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created factor_stability experiment directory: {experiment_output_dir}")

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
                output_dir=get_output_dir(exp_config),
            )
            logger.info(f"   Data loaded: {len(X_list)} views")

        # Extract view_names, feature_names, and subject_ids from preprocessing_info
        view_names = preprocessing_info.get("data_summary", {}).get("view_names", [f"view_{i}" for i in range(len(X_list))])
        feature_names = preprocessing_info.get("data_summary", {}).get("original_data", {}).get("feature_names", {})
        subject_ids = preprocessing_info.get("data_summary", {}).get("original_data", {}).get("subject_ids", None)
        logger.info(f"   Views: {view_names}")
        if subject_ids:
            logger.info(f"   Subject IDs: {len(subject_ids)} patients")

        # Prepare hyperparameters (use K_value which respects command-line override)
        # Read from global model section for consistency across pipeline
        model_config = exp_config.get("model", {})
        K = K_value
        hypers = {
            "Dm": [X.shape[1] for X in X_list],
            "K": K,
            "num_sources": len(X_list),  # Automatically infer from number of views
            "a_sigma": 1.0,
            "b_sigma": 1.0,
            "slab_df": model_config.get("slab_df", 4),
            "slab_scale": model_config.get("slab_scale", 2),
            "percW": model_config.get("percW", 33),
        }
        logger.info(f"   Using global model hyperparameters: percW={hypers['percW']}, slab_df={hypers['slab_df']}, slab_scale={hypers['slab_scale']}")
        logger.info(f"   Detected {hypers['num_sources']} data views/sources from X_list")

        # Prepare MCMC args (read from config.yaml)
        # Check both mcmc and factor_stability sections (mcmc takes precedence for command-line overrides)
        mcmc_config = exp_config.get("mcmc", {})
        max_tree_depth_value = mcmc_config.get("max_tree_depth") or fs_config.get("max_tree_depth", 10)
        target_accept_prob_value = mcmc_config.get("target_accept_prob") or fs_config.get("target_accept_prob", 0.8)

        mcmc_args = {
            "K": K,
            "num_warmup": fs_config.get("num_warmup", 1000),
            "num_samples": fs_config.get("num_samples", 5000),
            "num_chains": 1,  # Sequential execution (run_factor_stability_analysis handles multiple chains)
            "target_accept_prob": target_accept_prob_value,
            "max_tree_depth": max_tree_depth_value,
            "dense_mass": fs_config.get("dense_mass", False),
            "reghsZ": fs_config.get("reghsZ", True),
            "random_seed": 42,
            "model_type": model_config.get("model_type", "sparse_gfa"),  # Pass model type from config
            "use_pca_initialization": model_config.get("use_pca_initialization", False),  # PCA initialization
        }

        logger.info(f"   Using parameters from config.yaml:")
        logger.info(f"   - K={K}, percW={hypers['percW']}, slab_df={hypers['slab_df']}, slab_scale={hypers['slab_scale']}")
        logger.info(f"   - num_samples={mcmc_args['num_samples']}, num_warmup={mcmc_args['num_warmup']}")
        logger.info(f"   - target_accept_prob={mcmc_args['target_accept_prob']}, max_tree_depth={mcmc_args['max_tree_depth']}, dense_mass={mcmc_args['dense_mass']}")
        logger.info(f"   - cosine_threshold={experiment_config.cosine_threshold}, min_match_rate={experiment_config.min_match_rate}")

        # Run factor stability analysis
        logger.info(f"   Running {experiment_config.num_chains} chains with K={K} factors...")
        logger.info(f"   This will take approximately {experiment_config.num_chains * 10} minutes")

        try:
            result = repro_exp.run_factor_stability_analysis(
                X_list=X_list,
                hypers=hypers,
                args=mcmc_args,
                n_chains=experiment_config.num_chains,
                cosine_threshold=experiment_config.cosine_threshold,
                min_match_rate=experiment_config.min_match_rate,
                output_dir=str(experiment_output_dir),  # Pass experiment-specific output directory
                view_names=view_names,  # Pass view names for plotting
                feature_names=feature_names,  # Pass feature names for plotting
                subject_ids=subject_ids,  # Pass subject IDs for Z score indexing
            )
            logger.info("   ✅ Factor stability analysis COMPLETED")
        except Exception as e:
            logger.error(f"   ❌ Factor stability analysis FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result = None

        results["factor_stability"] = result
        logger.info(f"   Result type: {type(result)}")
        if result:
            logger.info(f"   Result status: {getattr(result, 'status', 'unknown')}")
            if hasattr(result, 'model_results'):
                logger.info(f"   Model results keys: {list(result.model_results.keys())}")

        # Save W and Z matrices for each chain + stability analysis results
        if result and hasattr(result, "model_results"):
            logger.info("   💾 Saving factor loadings and scores...")

            # Get output directory for factor stability
            if unified_dir:
                fs_output_dir = unified_dir / experiment_dir_mapping.get("factor_stability", "02_factor_stability")
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

            # Check if PCA was used in preprocessing
            preprocessor = preprocessing_info.get("preprocessing_results", {}).get("preprocessor")
            pca_enabled = preprocessing_info.get("preprocessing_results", {}).get("preprocessing_details", {}).get("pca_enabled", False)

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
                            view_name = view_names[view_idx] if view_idx < len(view_names) else f"view_{view_idx}"

                            # Save W_pcs (PC-space loadings) if PCA was used
                            if pca_enabled and preprocessor and hasattr(preprocessor, 'has_pca') and preprocessor.has_pca(view_name):
                                logger.info(f"      🔄 PCA-transformed view detected: {view_name}")

                                # This is in PC space - save as W_pcs
                                W_pcs_df = pd.DataFrame(
                                    W_view,
                                    columns=[f"Factor_{k}" for k in range(W_view.shape[1])],
                                )
                                W_pcs_df.index = [f"PC_{j}" for j in range(W_view.shape[0])]
                                W_pcs_df.index.name = "Component"
                                save_csv(W_pcs_df, chain_dir / f"W_pcs_view_{view_idx}.csv", index=True)
                                logger.info(f"      ✓ Saved W_pcs (PC space): {W_view.shape}")

                                # Transform back to voxel space for brain remapping
                                W_voxels = preprocessor.inverse_transform_pca_loadings(W_view, view_name)
                                if W_voxels is not None:
                                    W_voxels_df = pd.DataFrame(
                                        W_voxels,
                                        columns=[f"Factor_{k}" for k in range(W_voxels.shape[1])],
                                    )
                                    # Use feature names if available
                                    if feature_names and view_name in feature_names:
                                        view_feature_names = feature_names[view_name]
                                        if len(view_feature_names) == W_voxels.shape[0]:
                                            W_voxels_df.index = view_feature_names
                                        else:
                                            W_voxels_df.index = [f"Feature_{j}" for j in range(W_voxels.shape[0])]
                                    else:
                                        W_voxels_df.index = [f"Feature_{j}" for j in range(W_voxels.shape[0])]
                                    W_voxels_df.index.name = "Feature"
                                    save_csv(W_voxels_df, chain_dir / f"W_voxels_view_{view_idx}.csv", index=True)
                                    logger.info(f"      ✓ Saved W_voxels (voxel space): {W_voxels.shape}")
                                    logger.info(f"      📍 Brain remapping ready: use W_voxels_view_{view_idx}.csv with position lookup vectors")
                            else:
                                # No PCA - save directly as W (already in feature/voxel space)
                                W_df = pd.DataFrame(
                                    W_view,
                                    columns=[f"Factor_{k}" for k in range(W_view.shape[1])],
                                )
                                # Use feature names if available for this view
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
                Z = chain_data.get("Z")
                if Z is not None:
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
                    "chain_id": chain_data.get("chain_id", chain_id),
                    "seed": chain_data.get("seed", "unknown"),
                    "log_likelihood": float(chain_data.get("log_likelihood", 0)),
                    "convergence": chain_data.get("convergence", False),
                    "execution_time": chain_data.get("execution_time", 0),
                }
                save_json(metadata, chain_dir / "metadata.json", indent=2)

            # NOTE: save_stability_results() is already called inside run_factor_stability_analysis()
            # at experiments/robustness_testing.py:3208, so we skip the duplicate call here.
            # We only save the per-chain results and R-hat diagnostics which are unique to this section.

            # Save R-hat convergence diagnostics
            if hasattr(result, "diagnostics") and result.diagnostics:
                convergence_summary = result.diagnostics.get("convergence_summary", {})
                rhat_diagnostics = convergence_summary.get("rhat_diagnostics", {})
                if rhat_diagnostics:
                    rhat_file = fs_output_dir / "stability_analysis" / "rhat_convergence_diagnostics.json"
                    save_json(rhat_diagnostics, rhat_file, indent=2)
                    logger.info(f"   Saved R-hat convergence diagnostics to {rhat_file.name}")

            # Save plots
            if hasattr(result, "plots") and result.plots:
                plots_dir = fs_output_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                from core.io_utils import save_plot
                for plot_name, fig in result.plots.items():
                    if fig is not None:
                        save_plot(plots_dir / f"{plot_name}.png", dpi=300, close_after=False)
                        save_plot(plots_dir / f"{plot_name}.pdf", close_after=True)

            # Save PCA information README if PCA was used
            if pca_enabled and preprocessor:
                readme_path = fs_output_dir / "PCA_BRAIN_REMAPPING_README.md"
                with open(readme_path, "w") as f:
                    f.write("# PCA Dimensionality Reduction - Brain Remapping Guide\n\n")
                    f.write("This experiment used PCA dimensionality reduction during preprocessing.\n\n")
                    f.write("## Files Saved\n\n")
                    f.write("For each imaging view with PCA, two sets of factor loadings are saved:\n\n")
                    f.write("1. **W_pcs_view_X.csv**: Factor loadings in PC space (n_components × K)\n")
                    f.write("   - Direct output from Sparse GFA\n")
                    f.write("   - Rows: Principal components\n")
                    f.write("   - Columns: Latent factors\n\n")
                    f.write("2. **W_voxels_view_X.csv**: Factor loadings in voxel space (n_voxels × K)\n")
                    f.write("   - Transformed back through PCA: W_voxels = PCA.components_.T @ W_pcs\n")
                    f.write("   - Rows: Individual voxels (matched to position lookup)\n")
                    f.write("   - Columns: Latent factors\n")
                    f.write("   - **USE THIS FOR BRAIN REMAPPING IN MATLAB**\n\n")
                    f.write("## Brain Remapping Workflow\n\n")
                    f.write("```matlab\n")
                    f.write("% Load voxel-space loadings (already transformed)\n")
                    f.write("W_voxels = readmatrix('W_voxels_view_0.csv');\n\n")
                    f.write("% Load position lookup (use filtered version if available)\n")
                    f.write("positions = readtable('position_sn_voxels.tsv');\n\n")
                    f.write("% Multiply to create brain map for factor k\n")
                    f.write("k = 1;  % Factor index\n")
                    f.write("brain_map = positions;\n")
                    f.write("brain_map.loading = W_voxels(:, k);\n\n")
                    f.write("% Convert to NIfTI...\n")
                    f.write("```\n\n")
                    f.write("## PCA Details\n\n")
                    pca_info = preprocessing_info.get("preprocessing_results", {}).get("pca_info", {})
                    for view_name, info in pca_info.items():
                        f.write(f"### {view_name}\n")
                        f.write(f"- Original features: {info['n_features_original']}\n")
                        f.write(f"- PCA components: {info['n_components']}\n")
                        f.write(f"- Variance retained: {info['total_variance']:.2%}\n")
                        f.write(f"- Dimensionality reduction: {info['n_features_original']} → {info['n_components']} ")
                        f.write(f"({100 * info['n_components'] / info['n_features_original']:.1f}% of original)\n\n")
                logger.info(f"   ✓ Saved PCA brain remapping README: {readme_path}")

            logger.info(f"   ✓ Results saved to: {fs_output_dir}")

            # Log summary
            stability = result.model_results.get("stability_results", {})
            logger.info(f"   ✓ Found {stability.get('n_stable_factors', 0)}/{stability.get('total_factors', 0)} stable factors")
            logger.info(f"   ✓ Stability rate: {stability.get('stability_rate', 0):.1%}")
            if pca_enabled:
                logger.info(f"   ✓ Factor loadings saved in both PC space (W_pcs) and voxel space (W_voxels) for brain remapping")
            else:
                logger.info(f"   ✓ Factor loadings (W) and scores (Z) saved for all {len(chain_results_data)} chains")

            # Run comprehensive factor analysis diagnostics (Aspects 2, 4, 11, 16, 17, 18)
            factor_stability_config = config.get("factor_stability", {})
            diagnostics_config = factor_stability_config.get("diagnostics", {})
            run_diagnostics = diagnostics_config.get("run_all", True)  # Default: run all diagnostics

            if run_diagnostics and chain_results_data:
                logger.info("=" * 80)
                logger.info("🔬 Running comprehensive factor analysis diagnostics...")
                logger.info("=" * 80)

                try:
                    from analysis.factor_stability import (
                        assess_factor_stability_procrustes,
                        diagnose_scale_indeterminacy,
                        diagnose_slab_saturation,
                        diagnose_prior_posterior_shift,
                        classify_factor_types,
                        compute_cross_view_correlation,
                        save_diagnostic_results,
                    )
                    from visualization.diagnostic_plots import create_comprehensive_diagnostic_report

                    diagnostics = {}

                    # Aspect 2: Procrustes Alignment (Rotational Invariance)
                    logger.info("  📊 Aspect 2: Procrustes alignment...")
                    disparity_threshold = diagnostics_config.get("procrustes_alignment", {}).get("disparity_threshold", 0.3)
                    diagnostics['procrustes_alignment'] = assess_factor_stability_procrustes(
                        chain_results_data,
                        scale_normalize=True,
                        disparity_threshold=disparity_threshold
                    )
                    logger.info(f"    ✓ Alignment rate: {diagnostics['procrustes_alignment'].get('alignment_rate', 0):.1%}")

                    # Aspect 4: Scale Indeterminacy
                    logger.info("  📊 Aspect 4: Scale indeterminacy...")
                    diagnostics['scale_indeterminacy'] = diagnose_scale_indeterminacy(
                        chain_results_data,
                        tau_W_key='tauW',
                        tau_Z_key='tauZ'
                    )
                    logger.info(f"    ✓ Verdict: {diagnostics['scale_indeterminacy']['verdict']}")

                    # Aspect 11: Slab Saturation (CRITICAL for detecting preprocessing bugs)
                    logger.info("  📊 Aspect 11: Slab saturation...")
                    saturation_threshold = diagnostics_config.get("slab_saturation", {}).get("saturation_threshold", 0.8)
                    diagnostics['slab_saturation'] = diagnose_slab_saturation(
                        chain_results_data,
                        slab_scale=hypers.get('slab_scale', 2.0),
                        saturation_threshold=saturation_threshold
                    )
                    if diagnostics['slab_saturation']['issues']['data_preprocessing_failure']:
                        logger.error("    ❌ CRITICAL: Data preprocessing failure detected!")
                    else:
                        logger.info("    ✓ No preprocessing issues detected")

                    # Aspect 18: Prior-Posterior Shift
                    logger.info("  📊 Aspect 18: Prior-posterior shift...")
                    diagnostics['prior_posterior_shift'] = diagnose_prior_posterior_shift(
                        chain_results_data,
                        hypers=hypers
                    )
                    logger.info(f"    ✓ Overall: {diagnostics['prior_posterior_shift'].get('overall_classification', 'unknown')}")

                    # Aspect 16: Factor Type Classification (requires consensus W)
                    if stability_results_data.get("consensus_W") is not None:
                        logger.info("  📊 Aspect 16: Factor type classification...")
                        activity_threshold = diagnostics_config.get("factor_classification", {}).get("activity_threshold", 0.1)
                        diagnostics['factor_types'] = classify_factor_types(
                            stability_results_data["consensus_W"],
                            view_dims=hypers['Dm'],
                            activity_threshold=activity_threshold
                        )
                        summary = diagnostics['factor_types']['summary']
                        logger.info(f"    ✓ Shared: {summary['n_shared']}, View-specific: {summary['n_view_specific']}, Background: {summary['n_background']}")

                    # Aspect 17: Cross-View Correlation
                    logger.info("  📊 Aspect 17: Cross-view correlation...")
                    diagnostics['cross_view_correlation'] = compute_cross_view_correlation(
                        X_list,
                        view_names=view_names
                    )
                    logger.info(f"    ✓ Mean correlation: {diagnostics['cross_view_correlation']['summary']['mean_correlation']:.3f}")

                    # Save all diagnostics to CSV/JSON
                    diagnostics_dir = fs_output_dir / "diagnostics"
                    save_diagnostic_results(diagnostics, str(diagnostics_dir))

                    # Generate visualization plots
                    create_comprehensive_diagnostic_report(diagnostics, diagnostics_dir)

                    # Store diagnostics in result
                    if not hasattr(result, 'diagnostics') or result.diagnostics is None:
                        result.diagnostics = {}
                    result.diagnostics['factor_analysis_fundamentals'] = diagnostics

                    logger.info("=" * 80)
                    logger.info("✅ Factor analysis diagnostics completed successfully")
                    logger.info("=" * 80)

                except Exception as e:
                    logger.error(f"❌ Factor analysis diagnostics FAILED: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

            # Optional: Run PD subtype discovery using consensus factors
            # Check if pd_subtype_discovery is enabled in config
            factor_stability_config = config.get("factor_stability", {})
            if factor_stability_config.get("run_pd_subtype_discovery", False):
                logger.info("🧬 Running PD subtype discovery using consensus factors")

                try:
                    # Get consensus factors from stability results
                    stability_results_data = result.model_results.get("stability_results", {})
                    W_consensus = stability_results_data.get("consensus_W")
                    Z_consensus = stability_results_data.get("consensus_Z")

                    if W_consensus is None or Z_consensus is None:
                        logger.warning("   ⚠️  Consensus factors not available, skipping subtype discovery")
                    else:
                        logger.debug(f"   Consensus factors: W={W_consensus.shape}, Z={Z_consensus.shape}")

                        # Load clinical data for validation
                        try:
                            from data.qmap_pd import load_qmap_pd
                            qmap_data = load_qmap_pd(get_data_dir(config))

                            # Extract clinical measures for validation
                            clinical_data = {}
                            if hasattr(qmap_data, 'clinical_data') and qmap_data.clinical_data is not None:
                                clinical_data = {
                                    "updrs": qmap_data.clinical_data.get("updrs_total"),
                                    "moca": qmap_data.clinical_data.get("moca_total"),
                                    "age": qmap_data.clinical_data.get("age"),
                                    "sex": qmap_data.clinical_data.get("sex"),
                                    "disease_duration": qmap_data.clinical_data.get("disease_duration"),
                                }
                                # Remove None values
                                clinical_data = {k: v for k, v in clinical_data.items() if v is not None}
                                logger.debug(f"   Clinical data loaded: {list(clinical_data.keys())}")
                            else:
                                logger.debug("   No clinical data available for validation")
                        except Exception as e:
                            logger.warning(f"   Failed to load clinical data: {e}")
                            clinical_data = {}

                        # Convert empty dict to None for consistency
                        if not clinical_data:
                            clinical_data = None

                        # Initialize clinical validation experiment
                        from experiments.clinical_validation import ClinicalValidationExperiments
                        from experiments.framework import ExperimentConfig

                        exp_config_subtype = ExperimentConfig(
                            experiment_name="pd_subtype_discovery",
                            description="PD subtype discovery using consensus factors from stability analysis",
                            dataset="qmap_pd",
                            data_dir=get_data_dir(config),
                        )

                        clinical_exp = ClinicalValidationExperiments(exp_config_subtype, logger)

                        # Set output directory to factor_stability directory
                        if unified_dir:
                            subtype_output_dir = unified_dir / experiment_dir_mapping.get("factor_stability", "02_factor_stability") / "pd_subtype_discovery"
                        else:
                            subtype_output_dir = get_output_dir(config) / "factor_stability" / "pd_subtype_discovery"
                        subtype_output_dir.mkdir(parents=True, exist_ok=True)
                        clinical_exp.base_output_dir = subtype_output_dir

                        # Run subtype discovery directly on consensus factors
                        # We'll call the subtype_analyzer methods directly instead of re-running SGFA
                        logger.debug("   Discovering PD subtypes using clustering...")
                        subtype_results = clinical_exp.subtype_analyzer.discover_pd_subtypes(Z_consensus)

                        logger.info(f"   ✓ Discovered {subtype_results.get('optimal_k', 'N/A')} subtypes (silhouette={subtype_results.get('best_solution', {}).get('silhouette_score', 'N/A'):.3f})")

                        # Validate discovered subtypes against clinical measures
                        validation_results = None
                        if clinical_data is not None:
                            logger.debug("   Validating discovered subtypes against clinical measures...")
                            validation_results = clinical_exp.subtype_analyzer.validate_subtypes_clinical(
                                subtype_results, clinical_data, Z_consensus
                            )
                            logger.debug(f"   ✓ Clinical validation complete")

                        # Analyze factor patterns for each discovered subtype
                        logger.debug("   Analyzing factor patterns for each subtype...")
                        interpretation_results = clinical_exp.subtype_analyzer.analyze_subtype_factor_patterns(
                            Z_consensus,
                            subtype_results,
                            {"W": W_consensus, "Z": Z_consensus}
                        )

                        # Generate visualizations
                        logger.debug("   Generating subtype visualizations...")
                        from visualization.subtype_plots import PDSubtypeVisualizer
                        visualizer = PDSubtypeVisualizer(exp_config_subtype)

                        results_dict = {
                            "subtype_discovery": subtype_results,
                            "clinical_validation": validation_results,
                            "factor_interpretation": interpretation_results,
                        }

                        visualizer.create_pd_subtype_plots(
                            results_dict,
                            Z_consensus,
                            clinical_data,
                            subtype_output_dir
                        )

                        logger.info("   ✓ PD subtype visualization plots generated")

                        # Save subtype discovery results
                        from core.io_utils import save_json
                        save_json(
                            {
                                "subtype_discovery": subtype_results,
                                "clinical_validation": validation_results,
                                "factor_interpretation": interpretation_results,
                            },
                            subtype_output_dir / "subtype_discovery_results.json",
                            indent=2
                        )

                        logger.info(f"   ✓ Results saved to: {subtype_output_dir.name}/")

                except Exception as e:
                    logger.error(f"   ❌ PD subtype discovery failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

    if "clinical_validation" in experiments_to_run:
        logger.info("🏥 4/4 Starting Clinical Validation with Neuroimaging CV...")
        exp_config = config.copy()
        if pipeline_context["X_list"] is not None and use_shared_data:
            logger.info("   → Using shared data from previous experiments")

            # Extract view_names and feature_names from preprocessing_info
            preprocessing_info = pipeline_context.get("preprocessing_info", {})
            view_names = preprocessing_info.get("data_summary", {}).get(
                "view_names",
                [f"view_{i}" for i in range(len(pipeline_context["X_list"]))]
            )
            feature_names = preprocessing_info.get("data_summary", {}).get("original_data", {}).get("feature_names", {})
            logger.info(f"   Views: {view_names}")

            exp_config["_shared_data"] = {
                "X_list": pipeline_context["X_list"],
                "preprocessing_info": preprocessing_info,
                "view_names": view_names,  # Explicitly pass view_names
                "feature_names": feature_names,  # Explicitly pass feature_names
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
                "factor_stability": "02_factor_stability/        - Factor stability analysis across chains",
                "clinical_validation": "03_clinical_validation/ - Clinical validation studies",
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
