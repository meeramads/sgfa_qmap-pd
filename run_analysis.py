import argparse
import os
import time
import pickle
import sys
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import jax.random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.preprocessing import StandardScaler
#generate/load data
import get_data
#visualization module
import visualization
#logging
import logging
from loader_qmap_pd import load_qmap_pd as qmap_pd
from preprocessing import AdvancedPreprocessor, cross_validate_source_combinations

from utils import get_infparams, get_robustK

from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logging.info('Starting run_analysis.py')

# == CONDITIONAL IMPORTS ==
# Import CV module only when needed to avoid dependency issues
CV_AVAILABLE = False
try:
    from cross_validation import (
        SparseBayesianGFACrossValidator, 
        CVConfig
    )
    CV_AVAILABLE = True
    logging.info("Cross-validation module available")
except ImportError:
    logging.info("Cross-validation module not available - will run standard analysis only")

# Import weight-to-MRI module
try:
    from qmap_gfa_weight2mri_python import FactorToMRIMapper, integrate_with_visualization
    FACTOR_MAPPING_AVAILABLE = True
    logging.info("Factor-to-MRI mapping module available")
except ImportError:
    FACTOR_MAPPING_AVAILABLE = False
    logging.info("Factor-to-MRI mapping module not available")

# == UTILITY FUNCTIONS ==
def validate_and_setup_args(args):
    """
    Validate arguments and setup derived parameters with helpful error messages.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    args : argparse.Namespace
        Validated and potentially modified arguments
        
    Raises
    ------
    ValueError : For invalid parameter values
    RuntimeError : For incompatible parameter combinations
    """
    
    # Validate core model parameters
    if args.K <= 0:
        raise ValueError(f"Invalid number of factors K={args.K}. Must be positive integer (e.g., K=10).")
    
    if args.K > 100:
        logging.warning(f"Large number of factors K={args.K} may lead to computational issues.")
    
    if not (1 <= args.percW <= 100):
        raise ValueError(f"Invalid sparsity percentage percW={args.percW}. Must be between 1-100.")
    
    if args.num_samples <= 0:
        raise ValueError(f"Invalid num_samples={args.num_samples}. Must be positive integer.")
    
    if args.num_samples < 500:
        logging.warning(f"Small num_samples={args.num_samples} may lead to poor convergence. Consider ≥1000.")
    
    if args.num_warmup <= 0:
        raise ValueError(f"Invalid num_warmup={args.num_warmup}. Must be positive integer.")
    
    if args.num_chains <= 0:
        raise ValueError(f"Invalid num_chains={args.num_chains}. Must be positive integer.")
    
    if args.num_runs <= 0:
        raise ValueError(f"Invalid num_runs={args.num_runs}. Must be positive integer.")
    
    # Validate model type
    if args.model not in ['sparseGFA', 'GFA']:
        raise ValueError(f"Invalid model='{args.model}'. Must be 'sparseGFA' or 'GFA'.")
    
    # Validate dataset
    if args.dataset not in ['qmap_pd', 'synthetic']:
        raise ValueError(f"Invalid dataset='{args.dataset}'. Must be 'qmap_pd' or 'synthetic'.")
    
    # Check for conflicting parameter combinations
    if getattr(args, 'cv_only', False) and not CV_AVAILABLE:
        raise RuntimeError(
            "CV-only mode requested (--cv_only) but cross-validation module is not available. "
            "Please install cross-validation dependencies or run without --cv_only."
        )
    
    if getattr(args, 'nested_cv', False) and not getattr(args, 'run_cv', False) and not getattr(args, 'cv_only', False):
        logging.warning("--nested_cv specified but neither --run_cv nor --cv_only set. Enabling --run_cv.")
        args.run_cv = True
    
    if getattr(args, 'create_factor_maps', False) and not FACTOR_MAPPING_AVAILABLE:
        logging.warning(
            "Factor mapping requested but qmap_gfa_weight2mri_python module not available. "
            "Factor maps will be skipped."
        )
        args.create_factor_maps = False
    
    # Validate preprocessing parameters
    if getattr(args, 'enable_preprocessing', False):
        valid_imputation = ['median', 'mean', 'knn', 'iterative']
        if getattr(args, 'imputation_strategy', 'median') not in valid_imputation:
            raise ValueError(f"Invalid imputation_strategy. Must be one of: {valid_imputation}")
        
        valid_selection = ['variance', 'statistical', 'mutual_info', 'combined', 'none']
        if getattr(args, 'feature_selection', 'variance') not in valid_selection:
            raise ValueError(f"Invalid feature_selection. Must be one of: {valid_selection}")
        
        if getattr(args, 'n_top_features', None) is not None and args.n_top_features <= 0:
            raise ValueError(f"Invalid n_top_features={args.n_top_features}. Must be positive integer or None.")
        
        if not (0.0 <= getattr(args, 'missing_threshold', 0.1) <= 1.0):
            raise ValueError(f"Invalid missing_threshold. Must be between 0.0 and 1.0.")
        
        if getattr(args, 'variance_threshold', 0.0) < 0:
            raise ValueError(f"Invalid variance_threshold. Must be non-negative.")
    
    # Validate cross-validation parameters
    if hasattr(args, 'cv_folds') and getattr(args, 'cv_folds', 5) <= 1:
        raise ValueError(f"Invalid cv_folds={args.cv_folds}. Must be > 1.")
    
    if hasattr(args, 'cv_type'):
        valid_cv_types = ['standard', 'stratified', 'grouped', 'repeated']
        if getattr(args, 'cv_type', 'standard') not in valid_cv_types:
            raise ValueError(f"Invalid cv_type. Must be one of: {valid_cv_types}")
    
    # GPU availability check
    if args.device == 'gpu':
        try:
            import jax
            if not jax.devices('gpu'):
                logging.warning("GPU requested but not available. Falling back to CPU.")
                args.device = 'cpu'
        except Exception as e:
            logging.warning(f"JAX GPU support not available ({e}). Using CPU.")
            args.device = 'cpu'
    elif args.device not in ['cpu', 'gpu']:
        raise ValueError(f"Invalid device='{args.device}'. Must be 'cpu' or 'gpu'.")
    
    # Validate file paths for qMAP-PD
    if args.dataset == 'qmap_pd':
        from pathlib import Path
        data_path = Path(args.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data directory not found: {args.data_dir}. "
                f"Please ensure qMAP-PD data is available or use --dataset synthetic."
            )
        
        clinical_path = data_path / args.clinical_rel
        if not clinical_path.exists():
            raise FileNotFoundError(
                f"Clinical data file not found: {clinical_path}. "
                f"Please check the --clinical_rel parameter."
            )
        
        volumes_path = data_path / args.volumes_rel
        if not volumes_path.exists():
            raise FileNotFoundError(
                f"Volume matrices directory not found: {volumes_path}. "
                f"Please check the --volumes_rel parameter."
            )
    
    # Log important parameter combinations
    logging.info("=== VALIDATED PARAMETERS ===")
    logging.info(f"Model: {args.model}, Dataset: {args.dataset}")
    logging.info(f"Factors K={args.K}, Sparsity={args.percW}%, Samples={args.num_samples}")
    logging.info(f"Chains={args.num_chains}, Runs={args.num_runs}, Device={args.device}")
    
    if getattr(args, 'enable_preprocessing', False):
        logging.info(f"Preprocessing: {args.imputation_strategy} imputation, {args.feature_selection} selection")
    
    if getattr(args, 'run_cv', False) or getattr(args, 'cv_only', False):
        cv_folds = getattr(args, 'cv_folds', 5)
        cv_type = getattr(args, 'cv_type', 'standard')
        logging.info(f"Cross-validation: {cv_folds} folds, {cv_type} type")
        if getattr(args, 'nested_cv', False):
            logging.info("Nested CV enabled for hyperparameter optimization")
    
    logging.info("=" * 30)
    
    return args

# == MODEL CODE ==
def models(X_list, hypers, args):
    """Sparse GFA model with optional regularized horseshoe priors."""
    logging.debug(f"Running models with M={args.num_sources}, N={X_list[0].shape[0]}, Dm={list(hypers['Dm'])}")

    N, M = X_list[0].shape[0], args.num_sources
    Dm_np = np.array(hypers['Dm'], dtype=int)   
    Dm = jnp.array(Dm_np)                       
    assert len(X_list) == M, "Number of data sources does not match the number of provided datasets."
    for m in range(M):
        assert X_list[m].shape[0] == N, f"Data source {m+1} has inconsistent number of samples."
    D = int(Dm_np.sum())        
    K = args.K
    percW = hypers['percW']

    # Sample sigma
    sigma = numpyro.sample("sigma", dist.Gamma(hypers['a_sigma'], 
        hypers['b_sigma']),sample_shape=(1,M)) 
    
    if args.model == 'sparseGFA':
        
        # Sample Z   
        Z = numpyro.sample("Z",dist.Normal(0,1), sample_shape=(N,K))
        # Sample tau Z
        tauZ =  numpyro.sample(f'tauZ', dist.TruncatedCauchy(scale=1), sample_shape=(1,K)) 
        # Sample lambda Z
        lmbZ =  numpyro.sample("lmbZ", dist.TruncatedCauchy(scale=1), sample_shape=(N,K))
        if args.reghsZ:   
            # Sample cZ    
            cZtmp = numpyro.sample("cZ", dist.InverseGamma(0.5 * hypers['slab_df'], 
                0.5 * hypers['slab_df']), sample_shape=(1,K))
            cZ = hypers['slab_scale'] * jnp.sqrt(cZtmp)
            
            # Get regularised Z
            lmbZ_sqr = jnp.square(lmbZ)
            for k in range(K):    
                lmbZ_tilde = jnp.sqrt(lmbZ_sqr[:,k] * cZ[0,k] ** 2 / \
                    (cZ[0,k] ** 2 + tauZ[0,k] ** 2 * lmbZ_sqr[:,k]))
                Z = Z.at[:,k].set(Z[:,k] * lmbZ_tilde * tauZ[0,k])
        else:
            Z = Z * lmbZ * tauZ
    else:
        # Sample Z
        Z = numpyro.sample("Z",dist.Normal(0,1), sample_shape=(N,K))
    
    #sample W
    W = numpyro.sample("W",dist.Normal(0,1), sample_shape=(D,K))
    if 'sparseGFA' in args.model:
        # Implement regularised horseshoe prior over W
        #sample lambda W 
        lmbW =  numpyro.sample("lmbW", dist.TruncatedCauchy(scale=1), sample_shape=(D,K))
        #sample cW
        cWtmp = numpyro.sample("cW", dist.InverseGamma(0.5 * hypers['slab_df'], 
            0.5 * hypers['slab_df']), sample_shape=(M,K))
        cW = hypers['slab_scale'] * jnp.sqrt(cWtmp)
        pW = jnp.round((percW / 100.0) * Dm).astype(int)
        pW = jnp.clip(pW, 1, Dm - 1)

        d = 0
        for m in range(M): 
            X_m = jnp.asarray(X_list[m])
            scaleW = pW[m] / ((Dm[m] - pW[m]) * jnp.sqrt(N)) 
            #sample tau W
            tauW =  numpyro.sample(f'tauW{m+1}', 
                dist.TruncatedCauchy(scale=scaleW * 1/jnp.sqrt(sigma[0,m]))) 
            width = int(Dm_np[m])    
            lmbW_chunk = lax.dynamic_slice(lmbW, (d, 0), (width, K))
     
            lmbW_sqr = jnp.square(lmbW_chunk)
            lmbW_tilde = jnp.sqrt(cW[m,:] ** 2 * lmbW_sqr / 
                (cW[m,:] ** 2 + tauW ** 2 * lmbW_sqr))
            W_chunk = lax.dynamic_slice(W, (d, 0), (width, K))

            W_chunk = W_chunk * lmbW_tilde * tauW

            W = lax.dynamic_update_slice(W, W_chunk, (d, 0))
            #sample X
            W_chunk = lax.dynamic_slice(W, (d, 0), (width, K))

            numpyro.sample(f'X{m+1}', dist.Normal(jnp.dot(Z, W_chunk.T), 
                1/jnp.sqrt(sigma[0,m])), obs=X_m)
            d += width
    elif args.model == 'GFA':
        # Implement ARD prior over W
        alpha = numpyro.sample("alpha", dist.Gamma(1e-3, 1e-3), sample_shape=(M,K))
        d = 0
        for m in range(M): 
            X_m = jnp.asarray(X_list[m])
            
            width = int(Dm_np[m])    
            
            W_chunk = lax.dynamic_slice(W, (d, 0), (width, K))

            W_chunk = W_chunk * (1/jnp.sqrt(alpha[m,:]))

            W = lax.dynamic_update_slice(W, W_chunk, (d, 0))
            #sample X
            W_chunk = lax.dynamic_slice(W, (d, 0), (width, K))

            numpyro.sample(f'X{m+1}', dist.Normal(jnp.dot(Z, W_chunk.T), 
                1/jnp.sqrt(sigma[0,m])), obs=X_m)
            d += width
    
def run_inference(model, args, rng_key, X_list, hypers):
    
    # Run inference using Hamiltonian Monte Carlo
    kernel = NUTS(model, target_accept_prob=0.9, max_tree_depth=12)
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples, 
        num_chains=args.num_chains)
    mcmc.run(rng_key, X_list, hypers, args, extra_fields=('potential_energy',))    
    #mcmc.print_summary() 
    return mcmc

# == ORCHESTRATION FUNCTIONS ==

def run_cross_validation_analysis(args, X_list, hypers, data):
    """Orchestrate cross-validation analysis using cross_validation.py module."""
    if not CV_AVAILABLE:
        logging.error("Cross-validation requested but module not available!")
        logging.error("Make sure crossvalidation.py and its dependencies are installed")
        return None
    
    logging.info("=== ORCHESTRATING CROSS-VALIDATION ANALYSIS ===")
    
    # Setup CV configuration
    config = CVConfig()
    config.outer_cv_folds = getattr(args, 'cv_folds', 5)
    config.n_jobs = getattr(args, 'cv_n_jobs', 1)
    config.random_state = args.seed if args.seed else 42
    
    # Extract target/group variables if specified
    y = None
    groups = None
    
    if 'clinical' in data:
        clinical_df = data['clinical']
        
        if hasattr(args, 'cv_target_col') and args.cv_target_col and args.cv_target_col in clinical_df.columns:
            y = clinical_df[args.cv_target_col].values
            logging.info(f"Using {args.cv_target_col} for stratified CV")
        
        if hasattr(args, 'cv_group_col') and args.cv_group_col and args.cv_group_col in clinical_df.columns:
            groups = clinical_df[args.cv_group_col].values
            logging.info(f"Using {args.cv_group_col} for grouped CV")
    
    # Initialize cross-validator
    cv = SparseBayesianGFACrossValidator(config)
    
    # Determine CV type
    cv_type = getattr(args, 'cv_type', 'standard')
    
    # Run appropriate CV analysis
    if getattr(args, 'nested_cv', False):
        logging.info("Running nested cross-validation with hyperparameter optimization")
        search_space = {
            'K': [max(5, args.K//2), args.K, min(50, args.K*2)],
            'percW': [25, 33, 50],
            'num_samples': [max(500, args.num_samples//2), args.num_samples],
        }
        results = cv.nested_cross_validate(
            X_list, args, hypers, y, groups, cv_type, search_space
        )
    else:
        logging.info("Running standard cross-validation")
        results = cv.standard_cross_validate(
            X_list, args, hypers, y, groups, cv_type
        )
    
    return results, cv

def should_run_standard_analysis(args):
    """Determine if we should run standard MCMC analysis."""
    # Run standard analysis unless explicitly disabled or only CV requested
    return not getattr(args, 'cv_only', False)

def should_run_cv_analysis(args):
    """Determine if we should run cross-validation analysis."""
    return getattr(args, 'run_cv', False) or getattr(args, 'cv_only', False)

# == MAIN FUNCTION ==

def main(args):                           
    
    # === PARAMETER VALIDATION ===
    try:
        args = validate_and_setup_args(args)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        logging.error(f"Parameter validation failed: {e}")
        sys.exit(1)
    
    # === MEMORY CHECK ===
    from utils import check_available_memory, estimate_memory_requirements
    
    available_memory = check_available_memory()
    
    # Estimate memory for qMAP-PD (rough estimates)
    if args.dataset == 'qmap_pd':
        # These will be updated once data is loaded
        estimated_memory = estimate_memory_requirements(
            n_subjects=100,  # rough estimate
            n_features=2000,  # rough estimate  
            n_factors=args.K,
            n_chains=args.num_chains,
            n_samples=args.num_samples
        )
    
    # === RESULTS DIRECTORY SETUP ===
    from utils import create_results_structure, ensure_directory
    
    # Create flag strings
    if 'synthetic' in args.dataset: 
        flag = f'K{args.K}_{args.num_chains}chs_pW{args.percW}_s{args.num_samples}_addNoise{args.noise}'
    else:
        flag = f'K{args.K}_{args.num_chains}chs_pW{args.percW}_s{args.num_samples}'
    
    if args.model == 'sparseGFA':
        flag_regZ = '_reghsZ' if args.reghsZ else '_hsZ'
    else:
        flag_regZ = ''
    
    # Determine analysis modes
    run_standard = should_run_standard_analysis(args)
    run_cv = should_run_cv_analysis(args)
    
    logging.info(f"Analysis plan: Standard={run_standard}, CV={run_cv}")
    
    # Create result directories using Path
    results_base = Path('../results')
    
    if run_standard:
        standard_res_dir = create_results_structure(
            results_base, args.dataset, args.model, flag, flag_regZ
        )
        logging.info(f"Standard results directory: {standard_res_dir}")
    
    if run_cv:
        cv_res_dir = create_results_structure(
            results_base, f"{args.dataset}_cv", args.model, flag, f"{flag_regZ}_cv"
        )
        logging.info(f"CV results directory: {cv_res_dir}")

    # === HYPERPARAMETERS SETUP ===
    hp_dir = standard_res_dir if run_standard else cv_res_dir
    hp_path = hp_dir / 'hyperparameters.dictionary'
    
    from utils import safe_pickle_load, safe_pickle_save
    
    hypers = safe_pickle_load(hp_path, "Hyperparameters")
    if hypers is None:   
        hypers = {'a_sigma': 1, 'b_sigma': 1,
                'nu_local': 1, 'nu_global': 1,
                'slab_scale': 2, 'slab_df': 4, 
                'percW': args.percW}
        
        if not safe_pickle_save(hypers, hp_path, "Hyperparameters"):
            logging.error("Failed to save hyperparameters")
            return

    # === DATA LOADING (with memory monitoring) ===
    logging.info("=== LOADING AND PREPARING DATA ===")
    
    from utils import memory_monitoring_context
    
    with memory_monitoring_context("Data loading"):
        if 'synthetic' in args.dataset:  
            # Generate synthetic data
            data_path = hp_dir / 'synthetic_data.dictionary'
            data = safe_pickle_load(data_path, "Synthetic data")
            
            if data is None:
                data = get_data.synthetic_data(hypers, args)
                if not safe_pickle_save(data, data_path, "Synthetic data"):
                    logging.error("Failed to save synthetic data")
                    return
            
            # Convert synthetic data to multi-view format
            X = data['X'] 
            Dm = data['Dm']
            X_list = []
            d = 0
            for m in range(args.num_sources):
                X_list.append(X[:, d:d+Dm[m]])
                d += Dm[m]
            
            # Update hypers with Dm
            hypers.update({'Dm': Dm})
            
        elif 'qmap' in args.dataset:
            data = get_data.get_data(
                dataset=args.dataset,
                data_dir=args.data_dir,
                clinical_rel=args.clinical_rel,
                volumes_rel=args.volumes_rel,
                imaging_as_single_view=not args.roi_views,
                id_col=args.id_col,
                # Enhanced preprocessing parameters
                enable_advanced_preprocessing=getattr(args, 'enable_preprocessing', False),
                imputation_strategy=getattr(args, 'imputation_strategy', 'median'),
                feature_selection_method=getattr(args, 'feature_selection', 'variance'),
                n_top_features=getattr(args, 'n_top_features', None),
                missing_threshold=getattr(args, 'missing_threshold', 0.1),
                variance_threshold=getattr(args, 'variance_threshold', 0.0),
                target_variable=getattr(args, 'target_variable', None),
                cross_validate_sources=getattr(args, 'cross_validate_sources', False),
                optimize_preprocessing=getattr(args, 'optimize_preprocessing', False),
            ) 
            
            X_list = data['X_list']
            view_names = data['view_names']
            args.num_sources = len(X_list)
        
            logging.info(f"qMAP-PD views: {view_names} | Dm = {[x.shape[1] for x in X_list]}")
        
            # Log preprocessing results if available
            if 'preprocessing' in data:
                prep_results = data['preprocessing']
                logging.info("=== Preprocessing Applied ===")
                for view, stats in prep_results['feature_reduction'].items():
                    logging.info(f"{view}: {stats['original']} → {stats['processed']} features "
                           f"({stats['reduction_ratio']:.2%} retained)")
            
                if 'source_validation' in prep_results:
                    logging.info("Best source combinations by RMSE:")
                    sorted_combos = sorted(prep_results['source_validation'].items(), 
                                     key=lambda x: x[1]['rmse_mean'])
                    for combo, results in sorted_combos[:3]:  # Top 3
                        logging.info(f"  {combo}: {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
        
            # Update hypers with processed Dm
            hypers.update({'Dm': [x.shape[1] for x in X_list]})

            # Save preprocessing results if available
            if 'preprocessing' in data:
                prep_path = hp_dir / 'preprocessing_results.dictionary'
                if not safe_pickle_save(data['preprocessing'], prep_path, "Preprocessing results"):
                    logging.warning("Failed to save preprocessing results")

    # Update memory estimates with actual data
    if args.dataset == 'qmap_pd':
        n_subjects = X_list[0].shape[0]
        n_features = sum(X.shape[1] for X in X_list)
        
        logging.info(f"Actual data size: {n_subjects} subjects, {n_features} features")
        
        estimate_memory_requirements(
            n_subjects=n_subjects,
            n_features=n_features,
            n_factors=args.K,
            n_chains=args.num_chains,
            n_samples=args.num_samples
        )

    # === CROSS-VALIDATION ANALYSIS ===
    cv_results = None
    if run_cv:
        logging.info("=== RUNNING CROSS-VALIDATION ANALYSIS ===")
        try:
            with memory_monitoring_context("Cross-validation"):
                cv_result = run_cross_validation_analysis(args, X_list, hypers, data)
                if cv_result is not None:
                    cv_results, cv_object = cv_result
                    
                    # Save CV results
                    cv_object.save_results(cv_res_dir, "cv_analysis")
                    logging.info(f"CV results saved to {cv_res_dir}")
                    
                    # Create CV visualizations
                    try:
                        from visualization import plot_cv_results
                        plot_cv_results(cv_results, cv_res_dir, "cv_analysis")
                        logging.info("CV visualizations created")
                    except Exception as e:
                        logging.warning(f"Could not create CV visualizations: {e}")
                else:
                    logging.error("Cross-validation analysis failed")
        except Exception as e:
            logging.error(f"Cross-validation analysis failed: {e}")
            if getattr(args, 'cv_only', False):
                logging.error("CV-only mode requested but CV failed - exiting")
                return

    # === STANDARD MCMC ANALYSIS ===
    if run_standard:
        logging.info("=== RUNNING STANDARD MCMC ANALYSIS ===")
        
        from utils import get_model_files
        
        for i in range(args.num_runs):          
            logging.info(f'Initialisation: {i+1}')
            logging.info('----------------------------------')
            
            # Get file paths using utility function
            files = get_model_files(standard_res_dir, i+1)
            
            # RUN MODEL
            res_path = files['model_params']
            robparams_path = files['robust_params']

            # Run if file doesn't exist OR is empty
            if (not res_path.exists()) or (res_path.stat().st_size <= 5):
                try:
                    logging.info('Running MCMC Model...')
                    seed = np.random.randint(0, 50)
                    rng_key = jax.random.PRNGKey(seed)
                    
                    with memory_monitoring_context(f"MCMC run {i+1}"):
                        start = time.time()
                        MCMCout = run_inference(models, args, rng_key, X_list, hypers)
                        mcmc_samples = MCMCout.get_samples()

                        # Compute sampling performance
                        mcmc_samples.update({'time_elapsed': (time.time() - start)/60})
                        pe = MCMCout.get_extra_fields()['potential_energy']
                        mcmc_samples.update({'exp_logdensity': jnp.mean(-pe)})

                    # Save model only after success
                    if not safe_pickle_save(mcmc_samples, res_path, f"MCMC samples run {i+1}"):
                        logging.error(f"Failed to save MCMC samples for run {i+1}")
                        continue
                        
                    logging.info('Inferred parameters saved.')
                    
                except Exception as e:
                    logging.exception(f"MCMC run {i+1} failed")
                    # Clean up any corrupt file
                    if res_path.exists() and res_path.stat().st_size <= 5:
                        try:
                            res_path.unlink()
                        except OSError:
                            pass
                    continue  # move to next initialisation
                
            # Process robust parameters
            if (not robparams_path.exists()) and res_path.exists() and (res_path.stat().st_size > 5):    
                mcmc_samples = safe_pickle_load(res_path, f"MCMC samples run {i+1}")
                if mcmc_samples is None:
                    continue
                    
                inf_params, data_comps = get_infparams(mcmc_samples, hypers, args)

                if args.num_chains > 1:
                    #Find robust components
                    thrs = {'cosineThr': 0.8, 'matchThr': 0.5}
                    rob_params, X_rob, success = get_robustK(thrs, args, inf_params, data_comps)

                    #Save robust data components
                    if success:
                        rob_params.update({'sigma_inf': inf_params['sigma'], 'infX': X_rob})
                        if 'sparseGFA' in args.model:
                            rob_params.update({'tauW_inf': inf_params['tauW']}) 
                        
                        if not safe_pickle_save(rob_params, robparams_path, f"Robust parameters run {i+1}"):
                            logging.error(f"Failed to save robust parameters for run {i+1}")
                        else:
                            logging.info('Robust parameters saved')  
                    else:
                        logging.warning('No robust components found') 
                else:
                    W = np.mean(inf_params['W'][0], axis=0)
                    Z = np.mean(inf_params['Z'][0], axis=0)
                    X_recon = np.dot(Z, W.T)
                    rob_params = {'W': W, 'Z': Z, 'infX': X_recon}  
                    
                    if not safe_pickle_save(rob_params, robparams_path, f"Robust parameters run {i+1}"):
                        logging.error(f"Failed to save robust parameters for run {i+1}")

        # === VISUALIZATION ===
        with memory_monitoring_context("Visualization"):
            if 'synthetic' in args.dataset:
                # Need to reload synthetic data for visualization
                true_params = safe_pickle_load(standard_res_dir / 'synthetic_data.dictionary', "Synthetic data")
                if true_params:
                    visualization.synthetic_data(str(standard_res_dir), true_params, args, hypers)
            else:
                visualization.qmap_pd(data, str(standard_res_dir), args, hypers)

    # === COMPREHENSIVE VISUALIZATION ===
    if getattr(args, 'create_comprehensive_viz', False):
        logging.info("=== CREATING COMPREHENSIVE VISUALIZATION ===")
        try:
            with memory_monitoring_context("Comprehensive visualization"):
                from visualization import create_all_visualizations
                
                # Determine primary results directory
                primary_dir = standard_res_dir if run_standard else cv_res_dir
                
                # Create comprehensive visualization
                viz_args = {
                    'results_dir': str(primary_dir),
                    'data': data,
                    'run_name': f"comprehensive_{args.dataset}_{args.model}"
                }
                
                if cv_results:
                    viz_args['cv_results'] = cv_results
                
                create_all_visualizations(**viz_args)
                logging.info("Comprehensive visualization created")
                
        except Exception as e:
            logging.warning(f"Could not create comprehensive visualization: {e}")
            
    # === FACTOR-TO-MRI MAPPING ===
    if run_standard and getattr(args, 'create_factor_maps', False) and FACTOR_MAPPING_AVAILABLE:
        logging.info("=== CREATING FACTOR-TO-MRI MAPPINGS ===")
        try:
            with memory_monitoring_context("Factor-to-MRI mapping"):
                # Load best run results
                results_file = standard_res_dir / 'results.txt'
                brun = 1  # default
                
                if results_file.exists():
                    try:
                        with open(results_file, 'r') as f:
                            for line in f:
                                if line.startswith('Best run:'):
                                    brun = int(line.split(':')[1].strip())
                                    break
                    except:
                        pass
                
                # Load factor loadings
                files = get_model_files(standard_res_dir, brun)
                rob_params = safe_pickle_load(files['robust_params'], "Robust parameters")
                
                if rob_params and 'W' in rob_params:
                    W = rob_params['W']
                    
                    # Create factor maps
                    factor_maps = integrate_with_visualization(
                        str(standard_res_dir), data, W, args.data_dir,
                        factor_indices=list(range(min(10, W.shape[1])))  # Map first 10 factors
                    )
                    
                    logging.info("Factor-to-MRI mapping completed successfully")
                    logging.info(f"Generated {sum(len(maps) for maps in factor_maps.values())} NIfTI files")
                else:
                    logging.warning("No factor loadings found for mapping")
                    
        except Exception as e:
            logging.error(f"Factor-to-MRI mapping failed: {e}")

    # === FINAL CLEANUP AND SUMMARY ===
    from utils import cleanup_memory
    cleanup_memory()

    # === LOG FINAL SUMMARY ===
    logging.info("=" * 60)
    logging.info("ANALYSIS COMPLETE")
    logging.info("=" * 60)
    
    if run_standard:
        logging.info(f"Standard results: {standard_res_dir}")
    
    if run_cv and cv_results:
        logging.info(f"CV results: {cv_res_dir}")
        logging.info(f"CV Score: {cv_results.get('mean_cv_score', 'N/A'):.4f} ± {cv_results.get('std_cv_score', 'N/A'):.4f}")
    
    if run_standard and run_cv:
        logging.info("Both standard and CV analyses completed - compare results!")

if __name__ == "__main__":

    # Define arguments to run analysis
    dataset = 'qmap'
    if 'qmap' in dataset:
        num_samples = 5000
        K = 20
        num_sources = 2
        num_runs = 10
    else:
        num_samples = 1500
        K = 5
        num_sources = 3
        num_runs = 5

    parser = argparse.ArgumentParser(description=" Sparse GFA with reg. horseshoe priors")
    
    # == ORIGINAL ARGUMENTS (UNCHANGED) ==
    parser.add_argument("--model", nargs="?", default='sparseGFA', type=str, 
                        help='add horseshoe prior over the latent variables')
    parser.add_argument("--num-samples", nargs="?", default=num_samples, type=int, 
                        help='number of MCMC samples')
    parser.add_argument("--num-warmup", nargs='?', default=1000, type=int, 
                        help='number of MCMC samples for warmup')
    parser.add_argument("--K", nargs='?', default=K, type=int, 
                        help='number of components')
    parser.add_argument("--num-chains", nargs='?', default=4, type=int,
                        help= 'number of MCMC chains')
    parser.add_argument("--num-sources", nargs='?', default=num_sources, type=int, 
                        help='number of data sources')
    parser.add_argument("--num-runs", nargs='?', default=num_runs, type=int, 
                        help='number of runs')
    parser.add_argument("--reghsZ", nargs='?', default=True, type=bool)
    parser.add_argument("--percW", nargs='?', default=33, type=int, 
                        help='percentage of relevant variables in each source')
    parser.add_argument("--dataset", type=str, default="qmap_pd",
                        choices=["qmap_pd", "synthetic"])
    parser.add_argument("--data_dir", type=str, default="qMAP-PD_data")
    parser.add_argument("--device", default='cpu', type=str, 
                        help='use "cpu" or "gpu".')
    parser.add_argument("--noise", nargs='?', default=0, type=int, 
                        help='Add noise to synthetic data (1=yes, 0=no)')
    parser.add_argument("--seed", nargs='?', default=None, type=int,
                        help='Random seed for reproducibility (int). If not set, a random seed is used.')
    parser.add_argument("--clinical_rel", type=str, default="data_clinical/pd_motor_gfa_data.tsv")
    parser.add_argument("--volumes_rel", type=str, default="volume_matrices")
    parser.add_argument("--id_col", type=str, default="sid")
    parser.add_argument("--roi_views",action="store_true", 
                        help="If set, keep separate ROI views (SN/Putamen/Lentiform). If not set, concatenates imaging."
    )
    
    # == PREPROCESSING ARGUMENTS ==
    parser.add_argument("--enable_preprocessing", action="store_true",
                       help="Enable advanced preprocessing following Ferreira et al. and Bunte et al. methodologies")
    parser.add_argument("--imputation_strategy", type=str, 
                       choices=['median', 'mean', 'knn', 'iterative'], default='median',
                       help="Missing data imputation strategy")
    parser.add_argument("--feature_selection", type=str, 
                       choices=['variance', 'statistical', 'mutual_info', 'combined', 'none'], 
                       default='variance',
                       help="Feature selection method")
    parser.add_argument("--n_top_features", type=int, default=None,
                       help="Number of top features to select (None for threshold-based)")
    parser.add_argument("--missing_threshold", type=float, default=0.1,
                       help="Drop features with more than this fraction of missing values")
    parser.add_argument("--variance_threshold", type=float, default=0.0,
                       help="Drop features with variance below this threshold")
    parser.add_argument("--target_variable", type=str, default=None,
                       help="Clinical variable to use as target for supervised feature selection")
    parser.add_argument("--cross_validate_sources", action="store_true",
                       help="Cross-validate different source combinations")
    parser.add_argument("--optimize_preprocessing", action="store_true",
                       help="Optimize preprocessing parameters via cross-validation")

    # == CROSS-VALIDATION ARGUMENTS ==
    cv_group = parser.add_argument_group('Cross-Validation Options')
    cv_group.add_argument("--run_cv", action="store_true",
                         help="Run cross-validation analysis in addition to standard analysis")
    cv_group.add_argument("--cv_only", action="store_true",
                         help="Run ONLY cross-validation analysis (skip standard MCMC)")
    cv_group.add_argument("--cv_folds", type=int, default=5,
                         help="Number of cross-validation folds")
    cv_group.add_argument("--cv_type", type=str, default="standard",
                         choices=["standard", "stratified", "grouped", "repeated"],
                         help="Type of cross-validation")
    cv_group.add_argument("--nested_cv", action="store_true",
                         help="Use nested cross-validation for hyperparameter optimization")
    cv_group.add_argument("--cv_n_jobs", type=int, default=1,
                         help="Number of parallel jobs for CV")
    cv_group.add_argument("--cv_target_col", type=str, default=None,
                         help="Clinical variable for stratified cross-validation")
    cv_group.add_argument("--cv_group_col", type=str, default=None,
                         help="Clinical variable for grouped cross-validation")

    # == WEIGHT TO MRI INTEGRATION ARGUMENTS ==
    parser.add_argument("--create_factor_maps", action="store_true",
                       help="Create NIfTI files mapping factor loadings back to brain space")
    parser.add_argument("--factor_maps_dir", type=str, default="factor_maps",
                       help="Directory name for factor map outputs")
    parser.add_argument("--reference_mri", type=str, default=None,
                       help="Path to reference MRI (if different from standard location)")
    
    # == OUTPUT CONTROL ==
    parser.add_argument("--create_comprehensive_viz", action="store_true",
                       help="Create comprehensive visualization combining all analyses")

    args = parser.parse_args()
    
    # Set the seed for reproducibility
    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(0, 100000)
    np.random.seed(seed)
    
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    
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
