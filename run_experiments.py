#!/usr/bin/env python
"""
Remote Workstation Experiment Runner
Runs the complete experimental framework optimized for university GPU resources.

This modular version imports experiments from separate modules for better organization.
"""

from experiments.sgfa_parameter_comparison import run_sgfa_parameter_comparison
from experiments.sensitivity_analysis import run_sensitivity_analysis
from experiments.model_comparison import run_model_comparison
from experiments.data_validation import run_data_validation
from experiments.clinical_validation import run_clinical_validation
from experiments.reproducibility import run_reproducibility
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
            "sgfa_parameter_comparison",
            "model_comparison",
            "sensitivity_analysis",
            "clinical_validation",
            "reproducibility",
            "neuroimaging_hyperopt",
            "neuroimaging_cv_benchmarks",
            "all",
        ],
        default=["all"],
        help="Experiments to run",
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
        "--exclude-clinical",
        nargs="+",
        help="Exclude specific clinical features (e.g., --exclude-clinical age sex tiv)",
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
        if "data" not in config:
            config["data"] = {}
        config["data"]["select_rois"] = args.select_rois
        logger.info(f"Selecting ROIs: {args.select_rois}")

    # Configure clinical feature exclusion if provided
    if args.exclude_clinical:
        if "data" not in config:
            config["data"] = {}
        config["data"]["exclude_clinical_features"] = args.exclude_clinical
        logger.info(f"Excluding clinical features: {args.exclude_clinical}")

    # Setup unified results directory if requested
    if args.unified_results:
        # Create single timestamped directory for all experiments
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = get_output_dir(config)
        # Create directory name based on what's actually running
        if len(args.experiments) == 1:
            unified_dir = output_dir / f"{args.experiments[0]}_run_{run_timestamp}"
        elif len(args.experiments) < 5:
            unified_dir = output_dir / f"{'_'.join(args.experiments)}_run_{run_timestamp}"
        else:
            unified_dir = output_dir / f"complete_run_{run_timestamp}"
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
            "sgfa_parameter_comparison": "02_sgfa_parameter_comparison",
            "model_comparison": "03_model_comparison",
            "sensitivity_analysis": "04_sensitivity_analysis",
            "clinical_validation": "05_clinical_validation",
            "reproducibility": "06_reproducibility"
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
        experiments_to_run = [
            "data_validation",
            "model_comparison",
            "sgfa_parameter_comparison",
            "sensitivity_analysis",
            "clinical_validation",
            "reproducibility",
        ]

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

    if "sgfa_parameter_comparison" in experiments_to_run:
        logger.info("🔬 3/6 Starting SGFA Parameter Comparison Experiment...")
        exp_config = config.copy()
        if pipeline_context["X_list"] is not None and use_shared_data:
            logger.info("   → Using shared data from data_validation")
            exp_config["_shared_data"] = {
                "X_list": pipeline_context["X_list"],
                "preprocessing_info": pipeline_context["preprocessing_info"],
                "mode": "shared",
            }
        sgfa_result = run_sgfa_parameter_comparison(exp_config)
        results["sgfa_parameter_comparison"] = sgfa_result

        # Extract optimal parameters for downstream experiments
        if sgfa_result and use_shared_data and hasattr(sgfa_result, "get"):
            try:
                # Extract best performing variant info
                if "model_results" in sgfa_result:
                    model_results = sgfa_result["model_results"]
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
                            # Parse K and percW from variant name (e.g., "K5_percW25")
                            import re

                            match = re.match(r"K(\d+)_percW(\d+)", best_variant)
                            if match:
                                optimal_K = int(match.group(1))
                                optimal_percW = int(match.group(2))

                                pipeline_context["optimal_sgfa_params"] = {
                                    "K": optimal_K,
                                    "percW": optimal_percW,
                                    "variant_name": best_variant,
                                    "execution_time": best_score,
                                }

                                logger.info(
                                    f"🎯 Identified optimal SGFA parameters: {best_variant} ({ best_score:.1f}s)"
                                )

            except Exception as e:
                logger.warning(f"Could not extract optimal parameters: {e}")
                pipeline_context["optimal_sgfa_params"] = None

    if "model_comparison" in experiments_to_run:
        logger.info("🧠 2/6 Starting Model Architecture Comparison Experiment...")
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

    if "clinical_validation" in experiments_to_run:
        logger.info("🏥 5/6 Starting Clinical Validation with Neuroimaging CV...")
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

        results["clinical_validation"] = run_clinical_validation(exp_config)

    if "reproducibility" in experiments_to_run:
        logger.info("🔁 6/6 Starting Reproducibility Analysis...")
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

        results["reproducibility"] = run_reproducibility(exp_config)

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

        # Run neuroimaging hyperparameter optimization from sgfa_parameter_comparison
        from experiments.sgfa_parameter_comparison import SGFAParameterComparison
        from experiments.framework import ExperimentConfig

        experiment_config = ExperimentConfig.from_dict(exp_config)
        sgfa_exp = SGFAParameterComparison(experiment_config)

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
                if exp_name == "sgfa_parameter_comparison" and hasattr(
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
                "sgfa_parameter_comparison": "02_sgfa_parameter_comparison/   - SGFA hyperparameter optimization",
                "model_comparison": "03_model_comparison/   - Model architecture comparison",
                "sensitivity_analysis": "04_sensitivity_analysis/ - Parameter sensitivity studies",
                "clinical_validation": "05_clinical_validation/ - Clinical validation studies",
                "reproducibility": "06_reproducibility/     - Reproducibility and stability analysis"
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
                    if exp_name == "sgfa_parameter_comparison" and hasattr(
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
