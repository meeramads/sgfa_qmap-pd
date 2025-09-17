import argparse
import logging
import sys

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import lax
from numpyro.infer import MCMC, NUTS

import core.visualization as visualization

# Analysis modules
from analysis.config_manager import ConfigManager
from analysis.cross_validation import (
    CVRunner,
)
from analysis.data_manager import DataManager
from analysis.model_runner import ModelRunner

# Utilities
from core.utils import (
    check_available_memory,
    cleanup_memory,
    get_model_files,
    memory_monitoring_context,
    safe_pickle_load,
    validate_and_setup_args,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting run_analysis.py")


# == MODEL CODE (UNCHANGED) ==
def models(X_list, hypers, args):
    """Sparse GFA model with optional regularized horseshoe priors."""
    logging.debug(
        f"Running models with M={args.num_sources}, N={X_list[0].shape[0]}, Dm={list(hypers['Dm'])}"
    )

    N, M = X_list[0].shape[0], args.num_sources
    Dm_np = np.array(hypers["Dm"], dtype=int)
    Dm = jnp.array(Dm_np)
    assert (
        len(X_list) == M
    ), "Number of data sources does not match the number of provided datasets."
    for m in range(M):
        assert (
            X_list[m].shape[0] == N
        ), f"Data source {m + 1} has inconsistent number of samples."
    D = int(Dm_np.sum())
    K = args.K
    percW = hypers["percW"]

    # Sample sigma
    sigma = numpyro.sample(
        "sigma", dist.Gamma(hypers["a_sigma"], hypers["b_sigma"]), sample_shape=(1, M)
    )

    if args.model == "sparseGFA":

        # Sample Z
        Z = numpyro.sample("Z", dist.Normal(0, 1), sample_shape=(N, K))
        # Sample tau Z
        tauZ = numpyro.sample(
            f"tauZ", dist.TruncatedCauchy(scale=1), sample_shape=(1, K)
        )
        # Sample lambda Z
        lmbZ = numpyro.sample(
            "lmbZ", dist.TruncatedCauchy(scale=1), sample_shape=(N, K)
        )
        if args.reghsZ:
            # Sample cZ
            cZtmp = numpyro.sample(
                "cZ",
                dist.InverseGamma(0.5 * hypers["slab_df"], 0.5 * hypers["slab_df"]),
                sample_shape=(1, K),
            )
            cZ = hypers["slab_scale"] * jnp.sqrt(cZtmp)

            # Get regularised Z
            lmbZ_sqr = jnp.square(lmbZ)
            for k in range(K):
                lmbZ_tilde = jnp.sqrt(
                    lmbZ_sqr[:, k]
                    * cZ[0, k] ** 2
                    / (cZ[0, k] ** 2 + tauZ[0, k] ** 2 * lmbZ_sqr[:, k])
                )
                Z = Z.at[:, k].set(Z[:, k] * lmbZ_tilde * tauZ[0, k])
        else:
            Z = Z * lmbZ * tauZ
    else:
        # Sample Z
        Z = numpyro.sample("Z", dist.Normal(0, 1), sample_shape=(N, K))

    # sample W
    W = numpyro.sample("W", dist.Normal(0, 1), sample_shape=(D, K))
    if "sparseGFA" in args.model:
        # Implement regularised horseshoe prior over W
        # sample lambda W
        lmbW = numpyro.sample(
            "lmbW", dist.TruncatedCauchy(scale=1), sample_shape=(D, K)
        )
        # sample cW
        cWtmp = numpyro.sample(
            "cW",
            dist.InverseGamma(0.5 * hypers["slab_df"], 0.5 * hypers["slab_df"]),
            sample_shape=(M, K),
        )
        cW = hypers["slab_scale"] * jnp.sqrt(cWtmp)
        pW = jnp.round((percW / 100.0) * Dm).astype(int)
        pW = jnp.clip(pW, 1, Dm - 1)

        d = 0
        for m in range(M):
            X_m = jnp.asarray(X_list[m])
            scaleW = pW[m] / ((Dm[m] - pW[m]) * jnp.sqrt(N))
            # sample tau W
            tauW = numpyro.sample(
                f"tauW{m + 1}",
                dist.TruncatedCauchy(scale=scaleW * 1 / jnp.sqrt(sigma[0, m])),
            )
            width = int(Dm_np[m])
            lmbW_chunk = lax.dynamic_slice(lmbW, (d, 0), (width, K))

            lmbW_sqr = jnp.square(lmbW_chunk)
            lmbW_tilde = jnp.sqrt(
                cW[m, :] ** 2 * lmbW_sqr / (cW[m, :] ** 2 + tauW**2 * lmbW_sqr)
            )
            W_chunk = lax.dynamic_slice(W, (d, 0), (width, K))

            W_chunk = W_chunk * lmbW_tilde * tauW

            W = lax.dynamic_update_slice(W, W_chunk, (d, 0))
            # sample X
            W_chunk = lax.dynamic_slice(W, (d, 0), (width, K))

            numpyro.sample(
                f"X{m + 1}",
                dist.Normal(jnp.dot(Z, W_chunk.T), 1 / jnp.sqrt(sigma[0, m])),
                obs=X_m,
            )
            d += width
    elif args.model == "GFA":
        # Implement ARD prior over W
        alpha = numpyro.sample("alpha", dist.Gamma(1e-3, 1e-3), sample_shape=(M, K))
        d = 0
        for m in range(M):
            X_m = jnp.asarray(X_list[m])

            width = int(Dm_np[m])

            W_chunk = lax.dynamic_slice(W, (d, 0), (width, K))

            W_chunk = W_chunk * (1 / jnp.sqrt(alpha[m, :]))

            W = lax.dynamic_update_slice(W, W_chunk, (d, 0))
            # sample X
            W_chunk = lax.dynamic_slice(W, (d, 0), (width, K))

            numpyro.sample(
                f"X{m + 1}",
                dist.Normal(jnp.dot(Z, W_chunk.T), 1 / jnp.sqrt(sigma[0, m])),
                obs=X_m,
            )
            d += width


def run_inference(model, args, rng_key, X_list, hypers):
    """Run MCMC inference using Hamiltonian Monte Carlo"""
    kernel = NUTS(model, target_accept_prob=0.9, max_tree_depth=12)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
    )
    mcmc.run(rng_key, X_list, hypers, args, extra_fields=("potential_energy",))
    return mcmc


def main(args):
    """Main analysis function - now much cleaner and modular"""

    # Initialize configuration manager
    config_manager = ConfigManager(args)

    # Validate arguments with dependency information
    try:
        args = validate_and_setup_args(
            args,
            cv_available=config_manager.dependencies.cv_available,
            neuroimaging_cv_available=config_manager.dependencies.neuroimaging_cv_available,
            factor_mapping_available=config_manager.dependencies.factor_mapping_available,
            preprocessing_available=config_manager.dependencies.preprocessing_available,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        logging.error(f"Parameter validation failed: {e}")
        sys.exit(1)

    # Memory check
    try:
        available_gb = check_available_memory()
        if available_gb < 8:
            logging.error(
                "Insufficient memory (<8GB available). Analysis likely to fail."
            )
            if not getattr(args, "force_run", False):
                sys.exit(1)
        elif available_gb < 16:
            logging.warning(
                "Limited memory (<16GB). Consider using --quick_cv or reducing parameters."
            )
    except Exception as e:
        logging.warning(f"Memory check failed: {e}")

    # Setup analysis configuration and directories
    config = config_manager.setup_analysis_config()
    hypers = config_manager.setup_hyperparameters()

    # Load and prepare data
    data_manager = DataManager(args, config_manager.get_hyperparameters_dir())
    X_list, data, hypers = data_manager.load_and_prepare_data(hypers)
    data_manager.estimate_memory_requirements(X_list)

    # Run cross-validation analysis if requested
    cv_results = None
    if config.run_cv:
        cv_runner = CVRunner(args, config.cv_res_dir)
        cv_result = cv_runner.run_cv_analysis(X_list, hypers, data)
        if cv_result:
            cv_results, cv_object = cv_result
            _save_cv_results(
                cv_results, cv_object, config.cv_res_dir, config_manager.dependencies
            )

    # Run standard MCMC analysis if requested
    if config.run_standard:
        model_runner = ModelRunner(args, config.standard_res_dir)
        model_runner.run_standard_analysis(X_list, hypers)

        # Create visualizations
        _create_visualizations(args, config.standard_res_dir, data, hypers)

    # Create brain visualizations if requested
    if config.run_standard:
        _create_brain_visualizations(
            args, config.standard_res_dir, data, config_manager.dependencies
        )

    # Final cleanup and summary
    cleanup_memory()
    _log_final_summary(config, cv_results, config_manager.dependencies)


def _save_cv_results(cv_results, cv_object, cv_res_dir, dependencies):
    """Save cross-validation results"""
    try:
        if dependencies.neuroimaging_cv_available and hasattr(
            cv_object, "save_neuroimaging_results"
        ):
            cv_object.save_neuroimaging_results(cv_res_dir, "neuroimaging_cv_analysis")
        else:
            cv_object.save_results(cv_res_dir, "cv_analysis")

        logging.info(f"CV results saved to {cv_res_dir}")

        # Create CV visualizations
        try:
            from core.visualization import plot_cv_results

            plot_cv_results(cv_results, cv_res_dir, "cv_analysis")
            logging.info("CV visualizations created")
        except Exception as e:
            logging.warning(f"Could not create CV visualizations: {e}")

    except Exception as e:
        logging.error(f"Failed to save CV results: {e}")


def _create_visualizations(args, results_dir, data, hypers):
    """Create standard visualizations"""
    try:
        with memory_monitoring_context("Visualization"):
            if "synthetic" in args.dataset:
                true_params = safe_pickle_load(
                    results_dir / "synthetic_data.dictionary", "Synthetic data"
                )
                if true_params:
                    visualization.synthetic_data(
                        str(results_dir), true_params, args, hypers
                    )
            else:
                visualization.qmap_pd(data, str(results_dir), args, hypers)
    except Exception as e:
        logging.error(f"Visualization failed: {e}")


def _create_brain_visualizations(args, results_dir, data, dependencies):
    """Create brain-specific visualizations if requested"""
    if not (
        getattr(args, "create_factor_maps", False)
        or getattr(args, "create_subject_reconstructions", False)
        or getattr(args, "comprehensive_brain_viz", False)
    ):
        return

    if not dependencies.factor_mapping_available:
        logging.warning(
            "Brain visualization requested but factor mapping module not available"
        )
        return

    try:
        with memory_monitoring_context("Brain visualization"):
            from visualization.neuroimaging_utils import integrate_with_visualization

            # Load best run results
            results_file = results_dir / "results.txt"
            brun = 1  # default
            if results_file.exists():
                try:
                    with open(results_file, "r") as f:
                        for line in f:
                            if line.startswith("Best run:"):
                                brun = int(line.split(":")[1].strip())
                                break
                except BaseException:
                    pass

            # Load factor loadings
            files = get_model_files(results_dir, brun)
            rob_params = safe_pickle_load(files["robust_params"], "Robust parameters")

            if rob_params and "W" in rob_params:
                W = rob_params["W"]
                factor_maps = integrate_with_visualization(
                    str(results_dir),
                    data,
                    W,
                    args.data_dir,
                    factor_indices=list(range(min(10, W.shape[1]))),
                )
                logging.info("Brain visualization completed successfully")
            else:
                logging.warning("No factor loadings found for brain visualization")

    except Exception as e:
        logging.error(f"Brain visualization failed: {e}")


def _log_final_summary(config, cv_results, dependencies):
    """Log final analysis summary"""
    logging.info("=" * 60)
    logging.info("ANALYSIS COMPLETE")
    logging.info("=" * 60)

    if config.run_standard:
        logging.info(f"Standard results: {config.standard_res_dir}")

    if config.run_cv and cv_results:
        logging.info(f"CV results: {config.cv_res_dir}")

        if "mean_cv_score" in cv_results:
            cv_score = cv_results["mean_cv_score"]
            cv_std = cv_results.get("std_cv_score", 0)
            logging.info(f"CV Score: {cv_score:.4f} ± {cv_std:.4f}")

    if config.run_standard and config.run_cv:
        logging.info("Both standard and CV analyses completed - compare results!")


if __name__ == "__main__":

    # Define arguments to run analysis
    dataset = "qmap"
    if "qmap" in dataset:
        num_samples = 5000
        K = 20
        num_sources = 2
        num_runs = 10
    else:
        num_samples = 1500
        K = 5
        num_sources = 3
        num_runs = 5

    parser = argparse.ArgumentParser(
        description=" Sparse GFA with reg. horseshoe priors"
    )

    # == ORIGINAL ARGUMENTS (UNCHANGED) ==
    parser.add_argument(
        "--model",
        nargs="?",
        default="sparseGFA",
        type=str,
        help="add horseshoe prior over the latent variables",
    )
    parser.add_argument(
        "--num-samples",
        nargs="?",
        default=num_samples,
        type=int,
        help="number of MCMC samples",
    )
    parser.add_argument(
        "--num-warmup",
        nargs="?",
        default=1000,
        type=int,
        help="number of MCMC samples for warmup",
    )
    parser.add_argument(
        "--K", nargs="?", default=K, type=int, help="number of components"
    )
    parser.add_argument(
        "--num-chains", nargs="?", default=4, type=int, help="number of MCMC chains"
    )
    parser.add_argument(
        "--num-sources",
        nargs="?",
        default=num_sources,
        type=int,
        help="number of data sources",
    )
    parser.add_argument(
        "--num-runs", nargs="?", default=num_runs, type=int, help="number of runs"
    )
    parser.add_argument("--reghsZ", nargs="?", default=True, type=bool)
    parser.add_argument(
        "--percW",
        nargs="?",
        default=33,
        type=int,
        help="percentage of relevant variables in each source",
    )
    parser.add_argument(
        "--dataset", type=str, default="qmap_pd", choices=["qmap_pd", "synthetic"]
    )
    parser.add_argument("--data_dir", type=str, default="qMAP-PD_data")
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--noise",
        nargs="?",
        default=0,
        type=int,
        help="Add noise to synthetic data (1=yes, 0=no)",
    )
    parser.add_argument(
        "--seed",
        nargs="?",
        default=None,
        type=int,
        help="Random seed for reproducibility (int). If not set, a random seed is used.",
    )
    parser.add_argument(
        "--clinical_rel", type=str, default="data_clinical/pd_motor_gfa_data.tsv"
    )
    parser.add_argument("--volumes_rel", type=str, default="volume_matrices")
    parser.add_argument("--id_col", type=str, default="sid")
    parser.add_argument(
        "--roi_views",
        action="store_true",
        help="If set, keep separate ROI views (SN/Putamen/Lentiform). If not set, concatenates imaging.",
    )

    # == BASIC PREPROCESSING ARGUMENTS ==
    parser.add_argument(
        "--enable_preprocessing",
        action="store_true",
        help="Enable advanced preprocessing pipeline",
    )
    parser.add_argument(
        "--imputation_strategy",
        type=str,
        choices=["median", "mean", "knn", "iterative"],
        default="median",
        help="Missing data imputation strategy",
    )
    parser.add_argument(
        "--feature_selection",
        type=str,
        choices=["variance", "statistical", "mutual_info", "combined", "none"],
        default="variance",
        help="Feature selection method",
    )
    parser.add_argument(
        "--n_top_features",
        type=int,
        default=None,
        help="Number of top features to select (None for threshold-based)",
    )
    parser.add_argument(
        "--missing_threshold",
        type=float,
        default=0.1,
        help="Drop features with more than this fraction of missing values",
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        default=0.0,
        help="Drop features with variance below this threshold",
    )
    parser.add_argument(
        "--target_variable",
        type=str,
        default=None,
        help="Clinical variable to use as target for supervised feature selection",
    )
    parser.add_argument(
        "--cross_validate_sources",
        action="store_true",
        help="Cross-validate different source combinations",
    )
    parser.add_argument(
        "--optimize_preprocessing",
        action="store_true",
        help="Optimize preprocessing parameters via cross-validation",
    )

    # == NEUROIMAGING-SPECIFIC ARGUMENTS ==
    neuro_group = parser.add_argument_group("Neuroimaging-Specific Options")
    neuro_group.add_argument(
        "--enable_spatial_processing",
        action="store_true",
        help="Enable spatial processing for neuroimaging data",
    )
    neuro_group.add_argument(
        "--spatial_imputation",
        action="store_true",
        default=True,
        help="Use spatial neighbors for imputation",
    )
    neuro_group.add_argument(
        "--roi_based_selection",
        action="store_true",
        default=True,
        help="Use ROI-based feature selection instead of pure variance",
    )
    neuro_group.add_argument(
        "--harmonize_scanners", action="store_true", help="Apply scanner harmonization"
    )
    neuro_group.add_argument(
        "--scanner_info_col",
        type=str,
        default=None,
        help="Column name for scanner information in clinical data",
    )
    neuro_group.add_argument(
        "--qc_outlier_threshold",
        type=float,
        default=3.0,
        help="Threshold for outlier detection in quality control",
    )
    neuro_group.add_argument(
        "--spatial_neighbor_radius",
        type=float,
        default=5.0,
        help="Radius in mm for finding spatial neighbors",
    )
    neuro_group.add_argument(
        "--min_voxel_distance",
        type=float,
        default=3.0,
        help="Minimum distance in mm between selected voxels",
    )

    # == ENHANCED CROSS-VALIDATION ARGUMENTS ==
    cv_group = parser.add_argument_group("Cross-Validation Options")
    cv_group.add_argument(
        "--run_cv",
        action="store_true",
        help="Run cross-validation analysis in addition to standard analysis",
    )
    cv_group.add_argument(
        "--cv_only",
        action="store_true",
        help="Run ONLY cross-validation analysis (skip standard MCMC)",
    )
    cv_group.add_argument(
        "--neuroimaging_cv",
        action="store_true",
        help="Use neuroimaging-aware cross-validation (recommended for qMRI data)",
    )
    cv_group.add_argument(
        "--cv_folds", type=int, default=5, help="Number of cross-validation folds"
    )
    cv_group.add_argument(
        "--cv_type",
        type=str,
        default="standard",
        choices=["standard", "stratified", "grouped", "repeated"],
        help="Type of cross-validation (basic CV)",
    )
    cv_group.add_argument(
        "--neuro_cv_type",
        type=str,
        default="clinical_stratified",
        choices=["clinical_stratified", "site_aware", "standard"],
        help="Type of neuroimaging cross-validation",
    )
    cv_group.add_argument(
        "--nested_cv",
        action="store_true",
        help="Use nested cross-validation for hyperparameter optimization",
    )
    cv_group.add_argument(
        "--cv_n_jobs", type=int, default=1, help="Number of parallel jobs for CV"
    )
    cv_group.add_argument(
        "--cv_target_col",
        type=str,
        default=None,
        help="Clinical variable for stratified cross-validation",
    )
    cv_group.add_argument(
        "--cv_group_col",
        type=str,
        default=None,
        help="Clinical variable for grouped cross-validation",
    )
    cv_group.add_argument(
        "--quick_cv",
        action="store_true",
        help="Use reduced parameters for faster CV (for testing)",
    )

    # == BRAIN VISUALIZATION ARGUMENTS ==
    brain_group = parser.add_argument_group("Brain Visualization Options")
    brain_group.add_argument(
        "--create_subject_reconstructions",
        action="store_true",
        help="Create 3D NIfTI reconstructions of original subject data",
    )
    brain_group.add_argument(
        "--reconstruct_subjects",
        type=str,
        default=None,
        help="Comma-separated subject IDs to reconstruct (e.g., '1,5,10') or range '1-10'",
    )
    brain_group.add_argument(
        "--n_reconstruct",
        type=int,
        default=5,
        help="Number of subjects to reconstruct (default: 5, used if --reconstruct_subjects not specified)",
    )
    brain_group.add_argument(
        "--comprehensive_brain_viz",
        action="store_true",
        help="Create comprehensive brain visualization (factor maps + subject reconstructions)",
    )

    # == FACTOR-TO-MRI MAPPING ARGUMENTS ==
    mapping_group = parser.add_argument_group("Factor-to-MRI Mapping Options")
    mapping_group.add_argument(
        "--create_factor_maps",
        action="store_true",
        help="Create NIfTI files mapping factor loadings back to brain space",
    )
    mapping_group.add_argument(
        "--factor_maps_dir",
        type=str,
        default="factor_maps",
        help="Directory name for factor map outputs",
    )
    mapping_group.add_argument(
        "--reference_mri",
        type=str,
        default=None,
        help="Path to reference MRI (if different from standard location)",
    )

    # == OUTPUT CONTROL ==
    parser.add_argument(
        "--create_comprehensive_viz",
        action="store_true",
        help="Create comprehensive visualization combining all analyses",
    )

    args = parser.parse_args()

    # Handle quick_cv option
    if getattr(args, "quick_cv", False):
        args.num_samples = min(1000, args.num_samples)
        args.num_warmup = min(500, args.num_warmup)
        args.num_chains = 1
        args.cv_folds = 3
        logging.info("Quick CV mode: reduced parameters for faster execution")

    # Set the seed for reproducibility
    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(0, 100000)
    np.random.seed(seed)

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)


def run_sgfa_analysis(
    X_list,
    K=5,
    sparsity_level=0.3,
    num_samples=2000,
    num_warmup=1000,
    num_chains=1,
    output_dir="./results",
    **kwargs,
):
    """
    Simple wrapper function for running SGFA analysis.

    Parameters
    ----------
    X_list : List[np.ndarray]
        List of data matrices
    K : int, optional
        Number of factors
    sparsity_level : float, optional
        Sparsity level for factor loadings
    num_samples : int, optional
        Number of MCMC samples
    num_warmup : int, optional
        Number of warmup samples
    num_chains : int, optional
        Number of MCMC chains
    output_dir : str, optional
        Output directory
    **kwargs
        Additional arguments

    Returns
    -------
    dict
        Analysis results
    """
    import argparse
    from pathlib import Path

    # Create args object for main function
    args = argparse.Namespace()
    args.K = K
    args.percW = sparsity_level
    args.num_samples = num_samples
    args.num_warmup = num_warmup
    args.num_chains = num_chains
    args.output_dir = Path(output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set default args
    args.data_dir = kwargs.get("data_dir", "./data")
    args.num_grid_search = kwargs.get("num_grid_search", 0)
    args.num_cv_folds = kwargs.get("num_cv_folds", 5)
    args.disable_cv = kwargs.get("disable_cv", True)  # Disable CV for simple API
    args.cv_type = kwargs.get("cv_type", "standard")
    args.disable_visualization = kwargs.get("disable_visualization", False)
    args.disable_neuroimaging = kwargs.get("disable_neuroimaging", True)
    args.seed = kwargs.get("seed", 42)

    # Store X_list for the analysis
    # We'll need to save it temporarily and reference it
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save data temporarily
        for i, X in enumerate(X_list):
            np.save(temp_path / f"data_view_{i}.npy", X)

        args.data_dir = str(temp_path)
        args.load_synthetic_data = False

        try:
            # Run main analysis
            main(args)

            # Load and return results
            results_file = args.output_dir / "analysis_results.json"
            if results_file.exists():
                import json

                with open(results_file, "r") as f:
                    return json.load(f)
            else:
                return {"status": "completed", "output_dir": str(args.output_dir)}

        except Exception as e:
            logging.error(f"SGFA analysis failed: {e}")
            return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    # Error and stoppage handling
    try:
        main(args)
    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Analysis failed with error: {e}")
        logging.exception("Full traceback:")
        sys.exit(1)
