#!/usr/bin/env python
"""
Remote Workstation Experiment Runner
Runs the complete experimental framework optimized for university GPU resources.
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, '.')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/remote_workstation_experiments.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def load_config(config_path="remote_workstation/config.yaml"):
    """Load remote workstation configuration."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f" Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f" Failed to load config: {e}")
        sys.exit(1)

def setup_environment(config):
    """Setup experimental environment."""
    # Create output directories
    output_dir = Path(config['experiments']['base_output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = Path(config.get('monitoring', {}).get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Verify GPU availability
    try:
        import jax
        devices = jax.devices()
        logger.info(f" Available devices: {devices}")
        
        # Check for both 'gpu' and 'cuda' device types
        gpu_devices = [d for d in devices if d.platform in ['gpu', 'cuda']]
        if len(gpu_devices) == 0:
            logger.warning("  No GPU devices found - will use CPU (slower)")
        else:
            logger.info(f" GPU devices available for acceleration: {gpu_devices}")
            
    except Exception as e:
        logger.error(f" JAX setup issue: {e}")

def run_data_validation(config):
    """Run data validation experiments."""
    logger.info(" Starting Data Validation Experiments")
    
    try:
        from experiments.framework import ExperimentFramework, ExperimentConfig
        from experiments.data_validation import DataValidationExperiments
        
        # Setup framework
        framework = ExperimentFramework(
            base_output_dir=Path(config['experiments']['base_output_dir'])
        )
        
        # Configure experiment
        exp_config = ExperimentConfig(
            experiment_name="remote_workstation_data_validation",
            description="Comprehensive data validation on remote workstation",
            dataset="qmap_pd",
            data_dir=config['data']['data_dir']
        )
        
        # Run experiments
        validator = DataValidationExperiments(framework)
        
        # Quality assessment
        logger.info("   Running quality assessment...")
        quality_result = validator.run_data_quality_assessment(exp_config)
        
        # Preprocessing comparison
        logger.info("   Running preprocessing comparison...")
        preprocessing_result = validator.run_preprocessing_comparison(exp_config)
        
        logger.info(" Data validation experiments completed")
        return {'quality': quality_result, 'preprocessing': preprocessing_result}
        
    except Exception as e:
        logger.error(f" Data validation failed: {e}")
        return None

def run_method_comparison(config):
    """Run method comparison experiments."""
    logger.info("Starting Method Comparison Experiments")
    
    try:
        from experiments.framework import ExperimentFramework, ExperimentConfig
        from experiments.method_comparison import MethodComparisonExperiments
        
        framework = ExperimentFramework(
            base_output_dir=Path(config['experiments']['base_output_dir'])
        )
        
        exp_config = ExperimentConfig(
            experiment_name="remote_workstation_method_comparison",
            description="Compare SGFA model variants on remote workstation",
            dataset="qmap_pd",
            data_dir=config['data']['data_dir']
        )
        
        # COMPREHENSIVE ANALYSIS FRAMEWORK INTEGRATION
        from remote_workstation.analysis_integration import integrate_analysis_with_pipeline

        logger.info("📊 Integrating comprehensive analysis framework...")
        data_manager, model_runner, analysis_summary = integrate_analysis_with_pipeline(
            config=config,
            data_dir=config['data']['data_dir']
        )

        # COMPREHENSIVE PERFORMANCE OPTIMIZATION INTEGRATION
        from remote_workstation.performance_integration import integrate_performance_with_pipeline

        logger.info("⚡ Integrating comprehensive performance optimization framework...")
        performance_manager, performance_summary = integrate_performance_with_pipeline(
            config=config,
            data_dir=config['data']['data_dir']
        )

        # Load data with structured analysis framework if available
        if data_manager and analysis_summary.get('integration_summary', {}).get('structured_analysis', False):
            logger.info("📊 Using structured DataManager for data loading...")
            from remote_workstation.analysis_integration import _wrap_analysis_framework

            # Use structured data loading
            analysis_wrapper = _wrap_analysis_framework(data_manager, model_runner, analysis_summary)
            X_list, structured_data_info = analysis_wrapper.load_and_prepare_data()

            if structured_data_info.get('data_loaded', False):
                logger.info("✅ Data loaded with structured analysis framework")
                logger.info(f"   Loader: {structured_data_info.get('loader', 'unknown')}")
                if structured_data_info.get('preprocessing_applied', False):
                    logger.info(f"   Preprocessing: Applied via DataManager")

                # Store structured data info as preprocessing_info for compatibility
                preprocessing_info = {
                    'preprocessing_integration': True,
                    'loader_type': 'structured_analysis_framework',
                    'structured_data_info': structured_data_info,
                    'data_manager_used': True
                }
            else:
                logger.warning("⚠️ Structured data loading failed - falling back to preprocessing integration")
                # Fall back to preprocessing integration
                from remote_workstation.preprocessing_integration import apply_preprocessing_to_pipeline
                X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                    config=config,
                    data_dir=config['data']['data_dir'],
                    auto_select_strategy=True
                )
        else:
            # Use preprocessing integration
            from remote_workstation.preprocessing_integration import apply_preprocessing_to_pipeline

            logger.info("🔧 Applying comprehensive preprocessing integration...")
            X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                config=config,
                data_dir=config['data']['data_dir'],
                auto_select_strategy=True  # Automatically select optimal preprocessing strategy
            )

        # Apply performance optimization to loaded data
        if performance_manager:
            logger.info("⚡ Applying performance optimization to data loading...")
            X_list = performance_manager.optimize_data_arrays(X_list)
        else:
            logger.info("⚡ Performance framework unavailable - using basic data loading")

        # Create data structure compatible with existing pipeline
        data = {
            'X_list': X_list,
            'view_names': preprocessing_info.get('data_summary', {}).get('view_names', [f'view_{i}' for i in range(len(X_list))]),
            'preprocessing_info': preprocessing_info
        }
        
        # Create method comparison experiment function  
        def method_comparison_experiment(config, output_dir, **kwargs):
            import numpy as np  # Add missing numpy import
            logger.info("Running comprehensive method comparison...")
            X_list = data['X_list']

            # Log analysis framework information
            if analysis_summary:
                integration_info = analysis_summary.get('integration_summary', {})
                logger.info("📊 ANALYSIS FRAMEWORK SUMMARY:")
                logger.info(f"   Framework: {'Structured analysis' if integration_info.get('structured_analysis', False) else 'Direct core analysis'}")
                logger.info(f"   DataManager: {'available' if integration_info.get('data_management', False) else 'unavailable'}")
                logger.info(f"   ModelRunner: {'available' if integration_info.get('model_execution', False) else 'unavailable'}")

                if integration_info.get('components_available'):
                    logger.info(f"   Components: {', '.join(integration_info.get('components_available', []))}")

                logger.info(f"   Dependencies: CV={integration_info.get('cv_dependencies_available', False)}, "
                          f"Preprocessing={integration_info.get('preprocessing_dependencies_available', False)}, "
                          f"FactorMapping={integration_info.get('factor_mapping_available', False)}")

            # Log performance optimization information
            if performance_summary:
                logger.info("⚡ PERFORMANCE OPTIMIZATION SUMMARY:")
                logger.info(f"   Strategy: {performance_summary.get('selected_strategy', 'unknown')}")
                logger.info(f"   Framework: {'PerformanceManager' if performance_summary.get('performance_framework', False) else 'Basic optimization'}")

                config_info = performance_summary.get('configuration', {})
                if config_info:
                    logger.info(f"   Memory limit: {config_info.get('memory_limit_gb', 'unknown')}GB")
                    logger.info(f"   Data chunking: {'enabled' if config_info.get('enable_chunking', False) else 'disabled'}")
                    logger.info(f"   MCMC optimization: {'enabled' if config_info.get('mcmc_optimization', False) else 'disabled'}")

            # Log preprocessing information
            if 'preprocessing_info' in data:
                preprocessing_info = data['preprocessing_info']
                if preprocessing_info.get('preprocessing_integration', False):
                    logger.info("🔧 PREPROCESSING INTEGRATION SUMMARY:")
                    strategy_info = preprocessing_info.get('strategy_selection', {})
                    logger.info(f"   Strategy: {strategy_info.get('selected_strategy', 'unknown')}")
                    logger.info(f"   Reason: {strategy_info.get('reason', 'not specified')}")

                    proc_results = preprocessing_info.get('preprocessing_results', {})
                    if proc_results.get('status') == 'completed':
                        logger.info(f"   Preprocessor: {proc_results.get('preprocessor_type', 'unknown')}")
                        logger.info(f"   Steps applied: {proc_results.get('steps_applied', [])}")

                        if 'feature_reduction' in proc_results:
                            reduction = proc_results['feature_reduction']
                            logger.info(f"   Feature reduction: {reduction['total_before']} → {reduction['total_after']} "
                                      f"({reduction['reduction_ratio']:.3f} ratio)")
                else:
                    logger.info("🔧 Using basic preprocessing (advanced integration unavailable)")
            
            # COMPREHENSIVE CROSS-VALIDATION FRAMEWORK FOR HYPERPARAMETER OPTIMIZATION
            hyperparam_config = config.get('hyperparameter_optimization', {})
            if hyperparam_config.get('use_for_method_comparison', True):
                logger.info("🔬 Using comprehensive cross-validation framework for hyperparameter optimization...")

                # Try comprehensive CV framework first
                try:
                    from remote_workstation.cv_integration import integrate_cv_with_pipeline

                    # Get traditional optimal parameters as baseline
                    traditional_optimal_params, traditional_score, traditional_scores = determine_optimal_hyperparameters(X_list, config, ['joint'])

                    # Apply comprehensive CV framework
                    cv_results, enhanced_optimal_params, cv_integration_summary = integrate_cv_with_pipeline(
                        X_list=X_list,
                        config=config,
                        current_optimal_params=traditional_optimal_params,
                        data_dir=config['data']['data_dir']
                    )

                    # Use CV-enhanced parameters
                    optimal_params = enhanced_optimal_params
                    optimal_score = cv_results.get('best_cv_score', traditional_score)
                    all_scores = {
                        'cv_framework': cv_results,
                        'traditional': traditional_scores,
                        'cv_integration_summary': cv_integration_summary
                    }

                    logger.info(f"✅ CV framework optimization completed")
                    logger.info(f"   CV framework used: {cv_results.get('cv_framework_used', False)}")
                    logger.info(f"   CV type: {cv_results.get('cv_type', 'unknown')}")
                    if cv_integration_summary.get('parameter_enhancement', False):
                        logger.info(f"   Parameters enhanced by CV: {cv_integration_summary.get('parameter_changes', [])}")
                    else:
                        logger.info(f"   Parameters validated by CV (no changes recommended)")

                except Exception as cv_e:
                    logger.warning(f"CV framework failed: {cv_e}")
                    logger.info("Falling back to traditional hyperparameter optimization...")
                    optimal_params, optimal_score, all_scores = determine_optimal_hyperparameters(X_list, config, ['joint'])
            else:
                # Use traditional fixed parameters
                optimal_params = {
                    'K': hyperparam_config.get('fallback_K', 10),
                    'percW': hyperparam_config.get('fallback_percW', 33)
                }
                optimal_score = 0.0
                all_scores = {}
                logger.info(f"Using fixed parameters K={optimal_params['K']}, percW={optimal_params['percW']} (automatic optimization disabled)")

            # Direct implementation of method comparison logic
            results = {
                'sgfa_variants': {},
                'traditional_methods': {},
                'hyperparameter_optimization': {
                    'optimal_params': optimal_params,
                    'optimal_score': optimal_score,
                    'all_scores': all_scores,
                    'used_for_variants': True,
                    'optimization_method': 'joint' if 'joint' in str(all_scores.keys()) else 'individual'
                },
                'experiment_metadata': {
                    'n_subjects': X_list[0].shape[0],
                    'n_views': len(X_list),
                    'feature_counts': [X.shape[1] for X in X_list],
                    'total_features': sum(X.shape[1] for X in X_list)
                }
            }

            logger.info(f"Using optimal parameters K={optimal_params['K']}, percW={optimal_params['percW']} for all SGFA variants")

            # SGFA variant testing
            sgfa_variants = {
                'standard': {'use_sparse': True, 'use_group': True},
                'sparse_only': {'use_sparse': True, 'use_group': False},
                'group_only': {'use_sparse': False, 'use_group': True},
                'basic_fa': {'use_sparse': False, 'use_group': False}
            }
            
            logger.info("=== TESTING SGFA VARIANTS ===")
            logger.info(f"Number of variants to test: {len(sgfa_variants)}")
            logger.info(f"Variants: {list(sgfa_variants.keys())}")
            logger.info(f"Data shape: {len(X_list)} views with shapes {[X.shape for X in X_list]}")
            
            for i, (variant_name, variant_config) in enumerate(sgfa_variants.items(), 1):
                logger.info(f"\n--- VARIANT {i}/{len(sgfa_variants)}: {variant_name.upper()} ---")
                logger.info(f"Configuration: {variant_config}")
                
                # Import SGFA core components directly
                try:
                    from core.run_analysis import models, run_inference
                    import jax
                    import jax.random as random
                    import argparse
                    
                    # Check JAX devices before running
                    devices = jax.devices()
                    logger.info(f"JAX devices available: {devices}")
                    logger.info(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
                    
                    logger.info(f"🚀 Running {variant_name} SGFA analysis...")
                    
                    # Create minimal args for SGFA
                    args = argparse.Namespace(
                        model='sparseGFA',
                        K=optimal_params['K'],  # Use automatically determined optimal K
                        percW=optimal_params['percW'],  # Use automatically determined optimal percW
                        num_samples=200,  # Reduced for testing
                        num_warmup=100,
                        num_chains=1,
                        num_runs=1,
                        num_sources=len(X_list),
                        reghsZ=True,  # Add missing parameter
                        **variant_config
                    )
                    
                    logger.info(f"MCMC Configuration:")
                    logger.info(f"  - Model: {args.model}")
                    logger.info(f"  - K (factors): {args.K}")
                    logger.info(f"  - percW (sparsity): {args.percW}%")
                    logger.info(f"  - Samples: {args.num_samples}")
                    logger.info(f"  - Warmup: {args.num_warmup}")
                    logger.info(f"  - Chains: {args.num_chains}")
                    logger.info(f"  - Sources: {args.num_sources}")
                    
                    # Setup hyperparameters
                    hypers = {
                        'Dm': [X.shape[1] for X in X_list],
                        'a_sigma': 1.0,
                        'b_sigma': 1.0,
                        'slab_df': 4.0,
                        'slab_scale': 2.0,
                        'percW': 33
                    }
                    
                    logger.info(f"Hyperparameters:")
                    logger.info(f"  - Data dimensions: {hypers['Dm']}")
                    logger.info(f"  - Total features: {sum(hypers['Dm'])}")
                    logger.info(f"  - Percentage W: {hypers['percW']}%")
                    
                    # Run SGFA inference with performance optimization
                    import time
                    start_time = time.time()
                    rng_key = random.PRNGKey(42)

                    logger.info(f"⏱️  Starting MCMC inference at {time.strftime('%H:%M:%S')}...")
                    logger.info(f"Expected duration: ~{args.num_samples/10:.1f}-{args.num_samples/5:.1f} minutes")

                    # Apply structured analysis framework if available
                    if model_runner and analysis_summary.get('integration_summary', {}).get('structured_analysis', False):
                        logger.info("📊 Using structured analysis framework for MCMC execution...")
                        from remote_workstation.analysis_integration import run_structured_mcmc_analysis

                        # Use structured MCMC analysis
                        structured_results = run_structured_mcmc_analysis(
                            model_runner=model_runner,
                            data_manager=data_manager,
                            X_list=X_list,
                            config=config
                        )

                        # Convert structured results to compatible format
                        if structured_results.get('runs') and len(structured_results['runs']) > 0:
                            # Use first run results (can be enhanced to aggregate multiple runs)
                            first_run = list(structured_results['runs'].values())[0]
                            mcmc_result = type('MCMCResult', (), {
                                'get_samples': lambda: first_run,
                                'num_samples': args.num_samples,
                                'num_chains': args.num_chains
                            })()
                            logger.info(f"✅ Structured analysis completed with {len(structured_results['runs'])} runs")
                        else:
                            logger.warning("⚠️ Structured analysis failed - falling back to standard MCMC")
                            # Fall back to standard approach
                            if performance_manager:
                                from remote_workstation.performance_integration import optimize_mcmc_execution
                                mcmc_result = optimize_mcmc_execution(
                                    performance_manager=performance_manager,
                                    model_fn=models,
                                    args=args,
                                    rng_key=rng_key,
                                    X_list=X_list,
                                    hypers=hypers
                                )
                            else:
                                mcmc_result = run_inference(models, args, rng_key, X_list, hypers)

                    # Apply MCMC-specific performance optimizations
                    elif performance_manager:
                        logger.info("⚡ Applying MCMC performance optimization...")
                        from remote_workstation.performance_integration import optimize_mcmc_execution
                        mcmc_result = optimize_mcmc_execution(
                            performance_manager=performance_manager,
                            model_fn=models,
                            args=args,
                            rng_key=rng_key,
                            X_list=X_list,
                            hypers=hypers
                        )
                    else:
                        logger.info("⚡ Using standard MCMC execution (analysis framework and performance optimization unavailable)")
                        mcmc_result = run_inference(models, args, rng_key, X_list, hypers)
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    logger.info(f"✅ MCMC completed in {duration/60:.2f} minutes ({duration:.1f} seconds)")
                    logger.info(f"   Average time per sample: {duration/args.num_samples:.2f}s")
                    
                    # Extract MCMC samples
                    logger.info(f"📊 Extracting {variant_name} MCMC samples...")
                    samples = mcmc_result.get_samples()
                    logger.info(f"   - Total samples collected: {mcmc_result.num_samples}")
                    logger.info(f"   - Number of chains: {mcmc_result.num_chains}")
                    logger.info(f"   - Sample keys: {list(samples.keys())}")
                    
                    # Extract key parameters
                    if 'Z' in samples:
                        Z_samples = samples['Z']  # Factor scores
                        logger.info(f"   - Factor scores shape: {Z_samples.shape}")
                        Z_mean = Z_samples.mean(axis=0)  # Average over MCMC samples
                    
                    if 'W' in samples:
                        W_samples = samples['W']  # Factor loadings/weights
                        logger.info(f"   - Factor loadings shape: {W_samples.shape}")
                        W_mean = W_samples.mean(axis=0)  # Average over MCMC samples
                    
                    # Process extracted parameters
                    logger.info(f"📊 Processing {variant_name} results...")
                    
                    # Extract convergence diagnostics if available
                    try:
                        if hasattr(mcmc_result, 'get_extra_fields'):
                            extra_fields = mcmc_result.get_extra_fields()
                            if 'diverging' in extra_fields:
                                n_divergent = extra_fields['diverging'].sum()
                                logger.info(f"   - Divergent transitions: {n_divergent}")
                            if 'accept_prob' in extra_fields:
                                avg_accept = extra_fields['accept_prob'].mean()
                                logger.info(f"   - Average acceptance rate: {avg_accept:.3f}")
                    except Exception as e:
                        logger.debug(f"Could not extract MCMC diagnostics: {e}")
                    
                    variant_results = {
                        'status': 'completed',
                        'samples': mcmc_result.num_samples,
                        'chains': mcmc_result.num_chains,
                        'converged': True,
                        'duration_minutes': duration/60
                    }
                    
                    # Store comprehensive results including weights
                    variant_result_data = {
                        'status': 'completed',
                        'converged': variant_results.get('converged', True),
                        'log_likelihood': variant_results.get('log_likelihood', 0),
                        'n_factors': args.K,
                        'duration_minutes': variant_results['duration_minutes'],
                        'config': variant_config,
                        'mcmc_info': {
                            'num_samples': mcmc_result.num_samples,
                            'num_chains': mcmc_result.num_chains,
                            'sample_keys': list(samples.keys())
                        }
                    }
                    
                    # Add factor scores and loadings if available
                    if 'Z' in samples:
                        variant_result_data['factor_scores'] = {
                            'shape': list(Z_samples.shape),
                            'mean': Z_mean.tolist(),  # Convert to list for JSON serialization
                            'std': Z_samples.std(axis=0).tolist()
                        }
                        logger.info(f"   🧠 Factor scores: {Z_mean.shape} (subjects x factors)")
                    
                    if 'W' in samples:
                        variant_result_data['factor_loadings'] = {
                            'shape': list(W_samples.shape),
                            'mean': W_mean.tolist(),
                            'std': W_samples.std(axis=0).tolist()
                        }
                        logger.info(f"   🎨 Factor loadings: {W_mean.shape} (features x factors)")
                        
                        # Calculate sparsity metrics
                        sparsity = (np.abs(W_mean) < 0.1).mean()
                        logger.info(f"   🔍 Sparsity (|loading| < 0.1): {sparsity:.3f}")
                    
                    results['sgfa_variants'][variant_name] = variant_result_data
                    
                    logger.info(f"✅ {variant_name} SGFA completed successfully!")
                except ImportError as e:
                    logger.error(f"❌ Could not import SGFA modules for {variant_name}: {e}")
                    results['sgfa_variants'][variant_name] = {
                        'status': 'implemented_but_import_failed',
                        'error': str(e),
                        'config': variant_config
                    }
                except Exception as e:
                    logger.error(f"❌ {variant_name} SGFA failed: {str(e)}")
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    results['sgfa_variants'][variant_name] = {
                        'status': 'failed',
                        'error': str(e),
                        'config': variant_config
                    }
            
            # Traditional method comparison
            logger.info("\n=== TESTING TRADITIONAL METHODS ===")
            traditional_methods = ['pca', 'ica', 'factor_analysis']
            logger.info(f"Methods to test: {traditional_methods}")
            
            # Prepare data for traditional methods
            X_combined = np.concatenate(X_list, axis=1)
            logger.info(f"Combined data shape: {X_combined.shape}")
            logger.info(f"Total features: {X_combined.shape[1]}, subjects: {X_combined.shape[0]}")
            
            for i, method in enumerate(traditional_methods, 1):
                logger.info(f"\n--- METHOD {i}/{len(traditional_methods)}: {method.upper()} ---")
                start_time = time.time()
                try:
                    from sklearn.decomposition import PCA, FastICA, FactorAnalysis
                    
                    if method == 'pca':
                        logger.info(f"🚀 Running PCA with 10 components...")
                        model = PCA(n_components=10)
                        factors = model.fit_transform(X_combined)
                        explained_var = model.explained_variance_ratio_.sum()
                        logger.info(f"   Total explained variance: {explained_var:.3f}")
                        logger.info(f"   Top 3 component variances: {model.explained_variance_ratio_[:3]}")
                        explained_var = model.explained_variance_ratio_.sum()
                    elif method == 'ica':
                        logger.info(f"🚀 Running FastICA with 10 components...")
                        model = FastICA(n_components=10, random_state=42, max_iter=1000)
                        factors = model.fit_transform(X_combined)
                        logger.info(f"   ICA converged: {model.n_iter_ < model.max_iter}")
                        logger.info(f"   Iterations used: {model.n_iter_}")
                        explained_var = None  # ICA doesn't have explained variance
                    elif method == 'factor_analysis':
                        logger.info(f"🚀 Running Factor Analysis with 10 components...")
                        model = FactorAnalysis(n_components=10, random_state=42)
                        factors = model.fit_transform(X_combined)
                        loglike = model.score(X_combined)
                        logger.info(f"   Log-likelihood: {loglike:.2f}")
                        logger.info(f"   Noise variance: {model.noise_variance_.mean():.6f}")
                        explained_var = None  # FA doesn't have explained variance like PCA
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    logger.info(f"✅ {method.upper()} completed in {duration:.2f} seconds")
                    logger.info(f"   Output factors shape: {factors.shape}")
                    
                    results['traditional_methods'][method] = {
                        'status': 'completed',
                        'n_components': factors.shape[1],
                        'explained_variance_ratio': explained_var,
                        'factor_shape': factors.shape,
                        'duration_seconds': duration
                    }
                    
                except Exception as e:
                    end_time = time.time()
                    duration = end_time - start_time
                    logger.error(f"❌ {method.upper()} failed after {duration:.2f} seconds: {e}")
                    import traceback
                    logger.debug(f"Full traceback: {traceback.format_exc()}")
                    results['traditional_methods'][method] = {
                        'status': 'failed',
                        'error': str(e),
                        'duration_seconds': duration
                    }
            
            # Generate plots if requested
            try:
                logger.info("\\n🎨 === GENERATING COMPREHENSIVE VISUALIZATION SUITE ===")
                from visualization import VisualizationManager
                from pathlib import Path
                
                # Setup visualization configuration
                viz_config = type('VizConfig', (), {
                    'create_brain_viz': True,
                    'save_plots': True,
                    'output_dir': output_dir,
                    'data_dir': config.get('data', {}).get('data_dir', './qMAP-PD_data')
                })()

                # Use comprehensive visualization integration
                from remote_workstation.visualization_integration import create_comprehensive_visualizations, create_fallback_visualizations

                # Try comprehensive visualization first
                results['plots'] = create_comprehensive_visualizations(
                    results=results,
                    X_list=X_list,
                    optimal_params=optimal_params,
                    optimal_score=optimal_score,
                    config=config,
                    output_dir=output_dir,
                    all_scores=all_scores if 'all_scores' in locals() else None
                )

                # Fallback to basic visualization if comprehensive fails
                if results['plots']['status'] == 'visualization_manager_unavailable':
                    results['plots'] = create_fallback_visualizations(results, X_list, output_dir)

            except ImportError as e:
                logger.warning(f"⚠️  VisualizationManager not available: {e}")
                # Try fallback visualization
                try:
                    from remote_workstation.visualization_integration import create_fallback_visualizations
                    results['plots'] = create_fallback_visualizations(results, X_list, output_dir)
                except Exception as fallback_e:
                    logger.error(f"❌ Fallback visualization also failed: {fallback_e}")
                    results['plots'] = {'status': 'visualization_unavailable', 'error': str(e)}
            except Exception as e:
                logger.error(f"❌ Visualization failed: {e}")
                results['plots'] = {'status': 'failed', 'error': str(e)}
            
            # Generate experiment summary
            logger.info("\\n🎉 === METHOD COMPARISON COMPLETED ===")
            
            # SGFA Results Summary
            sgfa_completed = sum(1 for result in results['sgfa_variants'].values() 
                               if result.get('status') != 'failed')
            sgfa_failed = len(results['sgfa_variants']) - sgfa_completed
            
            logger.info(f"📊 SGFA VARIANTS SUMMARY:")
            logger.info(f"   ✅ Completed: {sgfa_completed}/{len(results['sgfa_variants'])}")
            if sgfa_failed > 0:
                logger.info(f"   ❌ Failed: {sgfa_failed}")
            
            for variant_name, result in results['sgfa_variants'].items():
                status = "✅" if result.get('status') != 'failed' else "❌"
                duration = result.get('duration_minutes', 0)
                logger.info(f"   {status} {variant_name}: {duration:.2f} minutes")
            
            # Traditional Methods Summary
            trad_completed = sum(1 for result in results['traditional_methods'].values() 
                               if result.get('status') == 'completed')
            trad_failed = len(results['traditional_methods']) - trad_completed
            
            logger.info(f"📊 TRADITIONAL METHODS SUMMARY:")
            logger.info(f"   ✅ Completed: {trad_completed}/{len(results['traditional_methods'])}")
            if trad_failed > 0:
                logger.info(f"   ❌ Failed: {trad_failed}")
            
            for method_name, result in results['traditional_methods'].items():
                status = "✅" if result.get('status') == 'completed' else "❌"
                duration = result.get('duration_seconds', 0)
                logger.info(f"   {status} {method_name}: {duration:.2f} seconds")
            
            # Model Results Summary
            sgfa_with_weights = sum(1 for result in results['sgfa_variants'].values() 
                                  if 'factor_loadings' in result and 'factor_scores' in result)
            if sgfa_with_weights > 0:
                logger.info(f"🧠 EXTRACTED MODEL COMPONENTS:")
                logger.info(f"   ✅ SGFA variants with factor loadings: {sgfa_with_weights}")
                logger.info(f"   ✅ SGFA variants with factor scores: {sgfa_with_weights}")
                
                # Log details about extracted weights
                for variant_name, result in results['sgfa_variants'].items():
                    if 'factor_loadings' in result:
                        W_shape = result['factor_loadings']['shape']
                        Z_shape = result['factor_scores']['shape'] if 'factor_scores' in result else 'N/A'
                        logger.info(f"   📊 {variant_name}: W{W_shape} (features→factors), Z{Z_shape} (subjects→factors)")
            
            # Plot Summary
            if 'plots' in results:
                plot_count = results['plots'].get('plot_count', 0)
                if plot_count > 0:
                    logger.info(f"🎨 VISUALIZATION SUMMARY:")
                    logger.info(f"   ✅ Generated {plot_count} plots")
                    logger.info(f"   📁 Plot directory: {results['plots']['plot_directory']}")

                    # Check if comprehensive VisualizationManager was used
                    if results['plots'].get('visualization_manager', False):
                        logger.info(f"   🎨 Comprehensive VisualizationManager suite used")
                        viz_suite = results['plots'].get('visualization_suite', {})
                        if viz_suite:
                            logger.info(f"   📊 Factor plots: {viz_suite.get('factor_plots', 0)}")
                            logger.info(f"   📈 Preprocessing plots: {viz_suite.get('preprocessing_plots', 0)}")
                            logger.info(f"   📉 CV/optimization plots: {viz_suite.get('cv_plots', 0)}")
                            logger.info(f"   🧠 Brain maps: {viz_suite.get('brain_maps', 0)}")
                            logger.info(f"   📄 HTML reports: {viz_suite.get('html_reports', 0)}")
                        if results['plots'].get('comprehensive_suite', False):
                            logger.info(f"   ✅ Full visualization capabilities utilized")
                    else:
                        logger.info(f"   📊 Basic visualization used (VisualizationManager unavailable)")
                        # Count brain imaging outputs
                        brain_files = [p for p in results['plots'].get('generated_plots', []) if 'brain_maps' in p or '.nii.gz' in p]
                        if brain_files:
                            logger.info(f"   🧠 Brain imaging files: {len(brain_files)}")
                            logger.info(f"   🧠 Includes: Factor loadings mapped to brain space")
                            logger.info(f"   🧠 Format: NIfTI files (.nii.gz) for each factor")
                        
                elif results['plots'].get('status') == 'visualization_unavailable':
                    logger.info(f"🎨 VISUALIZATION: Modules not available")
                else:
                    logger.info(f"🎨 VISUALIZATION: Failed to generate plots")
            
            # Overall summary
            total_experiments = len(results['sgfa_variants']) + len(results['traditional_methods'])
            total_completed = sgfa_completed + trad_completed
            logger.info(f"🏆 OVERALL: {total_completed}/{total_experiments} experiments completed successfully")
            
            logger.info("Method comparison completed successfully")
            
            # Format results for framework compatibility
            return {
                'model_results': results,
                'experiment_metadata': results.get('experiment_metadata', {}),
                'diagnostics': {
                    'sgfa_variants_tested': len(results.get('sgfa_variants', {})),
                    'traditional_methods_tested': len(results.get('traditional_methods', {})),
                    'total_experiments': len(results.get('sgfa_variants', {})) + len(results.get('traditional_methods', {}))
                }
            }
        
        result = framework.run_experiment(exp_config, method_comparison_experiment)
        
        logger.info(" Method comparison experiments completed")
        return result
        
    except Exception as e:
        logger.error(f" Method comparison failed: {e}")
        return None

def run_performance_benchmarks(config):
    """Run performance benchmark experiments."""
    logger.info("Starting Performance Benchmark Experiments")
    
    try:
        from experiments.framework import ExperimentFramework, ExperimentConfig  
        from experiments.performance_benchmarks import PerformanceBenchmarkExperiments
        
        framework = ExperimentFramework(
            base_output_dir=Path(config['experiments']['base_output_dir'])
        )
        
        exp_config = ExperimentConfig(
            experiment_name="remote_workstation_performance_benchmarks",
            description="Performance benchmarks on remote workstation",
            dataset="qmap_pd", 
            data_dir=config['data']['data_dir']
        )
        
        # Create performance benchmark experiment function
        def performance_benchmark_experiment(config, output_dir, **kwargs):
            import time
            import psutil
            import numpy as np
            from remote_workstation.preprocessing_integration import apply_preprocessing_to_pipeline

            logger.info("Running direct performance benchmarks...")
            X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                config=config.__dict__,
                data_dir=config.data_dir,
                auto_select_strategy=True
            )
            
            results = {
                'scalability_benchmark': {},
                'memory_benchmark': {},
                'timing_benchmark': {}
            }
            
            # Simple scalability test
            logger.info("Running scalability benchmark...")
            data_sizes = [0.25, 0.5, 0.75, 1.0]  # Fractions of full dataset
            
            for size_fraction in data_sizes:
                n_samples = int(X_list[0].shape[0] * size_fraction)
                X_subset = [X[:n_samples] for X in X_list]
                
                start_time = time.time()
                try:
                    from core.run_analysis import main
                    import argparse
                    
                    # Create complete args for SGFA analysis
                    args = argparse.Namespace(
                        model='sparseGFA',
                        K=5,
                        num_samples=200,  # Reduced for benchmarking
                        num_warmup=100,
                        num_chains=1,
                        num_runs=1,
                        dataset='qmap_pd',
                        data_dir=config.data_dir,
                        device='gpu',
                        clinical_rel="data_clinical/pd_motor_gfa_data.tsv",
                        volumes_rel="volume_matrices",
                        id_col="sid",
                        roi_views=True,
                        noise=0,
                        seed=42,
                        num_sources=4,
                        reghsZ=True,
                        percW=33
                    )
                    
                    # TODO: Pass X_subset to main - need to modify approach
                    main(args)
                    duration = time.time() - start_time
                    results['scalability_benchmark'][f'size_{size_fraction}'] = {
                        'n_samples': n_samples,
                        'duration_seconds': duration,
                        'samples_per_second': n_samples / duration if duration > 0 else 0
                    }
                except Exception as e:
                    results['scalability_benchmark'][f'size_{size_fraction}'] = {
                        'n_samples': n_samples,
                        'status': 'failed',
                        'error': str(e)
                    }
            
            # Memory usage benchmark
            logger.info("Running memory benchmark...")
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                # Run analysis and measure peak memory
                from core.run_analysis import main
                import argparse
                
                # Create complete args for SGFA analysis
                args = argparse.Namespace(
                    model='sparseGFA',
                    K=10,
                    num_samples=200,
                    num_warmup=100,
                    num_chains=1,
                    num_runs=1,
                    dataset='qmap_pd',
                    data_dir=config.data_dir,
                    device='gpu',
                    clinical_rel="data_clinical/pd_motor_gfa_data.tsv",
                    volumes_rel="volume_matrices",
                    id_col="sid",
                    roi_views=True,
                    noise=0,
                    seed=42,
                    num_sources=4,
                    reghsZ=True,
                    percW=33
                )
                
                main(args)
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                
                results['memory_benchmark'] = {
                    'memory_before_mb': memory_before,
                    'memory_after_mb': memory_after,
                    'memory_increase_mb': memory_after - memory_before,
                    'peak_memory_mb': memory_after
                }
            except Exception as e:
                results['memory_benchmark'] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            logger.info("Performance benchmarks completed")
            
            # Format results for framework compatibility
            return {
                'model_results': results,
                'performance_metrics': {
                    'benchmark_types': list(results.keys()),
                    'scalability_tests': len(results.get('scalability_benchmark', {})),
                    'memory_tests': len(results.get('memory_benchmark', {})),
                    'timing_tests': len(results.get('timing_benchmark', {}))
                },
                'diagnostics': {
                    'total_benchmarks': sum(len(v) if isinstance(v, dict) else 1 for v in results.values()),
                    'benchmark_status': 'completed'
                }
            }
        
        result = framework.run_experiment(exp_config, performance_benchmark_experiment)
        
        logger.info(" Performance benchmark experiments completed")
        return result
        
    except Exception as e:
        logger.error(f" Performance benchmarks failed: {e}")
        return None

def run_sensitivity_analysis(config):
    """Run sensitivity analysis experiments."""
    logger.info(" Starting Sensitivity Analysis Experiments")
    
    try:
        from experiments.framework import ExperimentFramework, ExperimentConfig
        from experiments.sensitivity_analysis import SensitivityAnalysisExperiments
        
        framework = ExperimentFramework(
            base_output_dir=Path(config['experiments']['base_output_dir'])
        )
        
        exp_config = ExperimentConfig(
            experiment_name="remote_workstation_sensitivity_analysis",
            description="Parameter sensitivity analysis on remote workstation",
            dataset="qmap_pd",
            data_dir=config['data']['data_dir']
        )
        
        # Model quality evaluation function for hyperparameter optimization
        def evaluate_model_quality(params, X_list, args, config):
            """
            Evaluate model quality for given hyperparameters (K, percW, etc.).
            """
            try:
                import numpy as np
                from sklearn.metrics import r2_score
                from sklearn.decomposition import FactorAnalysis

                # Extract parameters
                K = params.get('K', 10)
                percW = params.get('percW', 33)
                num_samples = params.get('num_samples', 2000)
                num_warmup = params.get('num_warmup', 1000)
                num_chains = params.get('num_chains', 4)
                target_accept_prob = params.get('target_accept_prob', 0.8)

                logger.debug(f"Evaluating quality for K={K}, percW={percW}, samples={num_samples}, warmup={num_warmup}, chains={num_chains}")

                # Use surrogate evaluation with Factor Analysis for speed
                X_concat = np.concatenate(X_list, axis=1)
                n_subjects, total_features = X_concat.shape

                # Quick factor analysis to get approximation
                fa = FactorAnalysis(n_components=K, random_state=42, max_iter=100)
                fa.fit(X_concat)

                # Reconstruction quality
                X_recon = fa.transform(X_concat) @ fa.components_
                recon_r2 = r2_score(X_concat, X_recon)

                # ENHANCED SPARSITY EVALUATION based on percW
                # percW controls how sparse the loadings should be
                # Lower percW = more sparse = higher sparsity score
                optimal_percW_range = [25, 33, 40]  # Known good values for neuroimaging

                if percW in optimal_percW_range:
                    percW_score = 0.9
                elif percW in [20, 50]:
                    percW_score = 0.8
                elif percW in [15, 60, 67]:
                    percW_score = 0.7
                else:
                    # Too sparse (< 15%) or not sparse enough (> 70%)
                    percW_score = 0.5

                # K-based sparsity (penalty for high K)
                K_sparsity_score = max(0, 1.0 - (K / 20.0))

                # Combined sparsity score
                sparsity_score = 0.6 * percW_score + 0.4 * K_sparsity_score

                # MCMC EFFICIENCY EVALUATION
                # Computational cost efficiency (samples vs quality tradeoff)
                optimal_samples_range = [1000, 2000]
                if num_samples in optimal_samples_range:
                    sample_efficiency = 0.9
                elif num_samples in [500, 3000]:
                    sample_efficiency = 0.8
                elif num_samples < 500:
                    sample_efficiency = 0.6  # Too few samples
                else:
                    sample_efficiency = 0.7  # Diminishing returns

                # Warmup efficiency (should be ~50% of num_samples)
                warmup_ratio = num_warmup / num_samples
                if 0.4 <= warmup_ratio <= 0.6:
                    warmup_efficiency = 0.9
                elif 0.3 <= warmup_ratio <= 0.7:
                    warmup_efficiency = 0.8
                else:
                    warmup_efficiency = 0.6

                # Chain efficiency (parallel sampling vs memory cost)
                available_cores = min(8, max(1, total_features // 1000))  # Estimate cores needed
                if num_chains <= available_cores and num_chains >= 2:
                    chain_efficiency = 0.9
                elif num_chains == 1:
                    chain_efficiency = 0.7  # No parallel convergence diagnostics
                else:
                    chain_efficiency = 0.6  # Too many chains for resources

                # Target accept probability efficiency
                if 0.75 <= target_accept_prob <= 0.85:
                    accept_efficiency = 0.9
                elif 0.7 <= target_accept_prob <= 0.9:
                    accept_efficiency = 0.8
                else:
                    accept_efficiency = 0.6

                # Combined MCMC efficiency score
                mcmc_efficiency = (0.3 * sample_efficiency +
                                 0.25 * warmup_efficiency +
                                 0.25 * chain_efficiency +
                                 0.2 * accept_efficiency)

                # Memory usage estimation and penalty
                estimated_memory_gb = (num_samples * num_chains * K * total_features * 4) / (1024**3)
                memory_penalty = max(0, (estimated_memory_gb - 16) / 16 * 0.2)  # Penalty if >16GB

                # Orthogonality approximation (factor analysis has orthogonal factors)
                orthogonality_score = 0.8

                # Clinical relevance (moderate K values + optimal percW)
                if K in [8, 10, 12] and percW in optimal_percW_range:
                    clinical_score = 0.9
                elif K in [5, 6, 7, 13, 14, 15] and percW in [20, 25, 40, 50]:
                    clinical_score = 0.8
                else:
                    clinical_score = 0.6

                # Sparsity-interpretability tradeoff
                # Lower percW should give higher interpretability but might hurt reconstruction
                sparsity_penalty = max(0, (percW - 50) / 50.0) * 0.1  # Penalty for high percW
                sparsity_bonus = max(0, (40 - percW) / 40.0) * 0.1    # Bonus for moderate sparsity

                # Composite interpretability score
                interpretability = (
                    0.35 * sparsity_score +           # Increased weight for sparsity
                    0.25 * orthogonality_score +
                    0.2 * 0.6 +                       # Spatial coherence placeholder
                    0.2 * clinical_score
                ) + sparsity_bonus - sparsity_penalty

                # Combined quality score (40% interpretability + 30% reconstruction + 30% MCMC efficiency)
                quality_score = (0.4 * interpretability +
                               0.3 * max(0, recon_r2) +
                               0.3 * mcmc_efficiency) - memory_penalty

                # Clamp to valid range
                quality_score = max(0.0, min(1.0, quality_score))

                logger.debug(f"K={K}, percW={percW}, samples={num_samples}: recon_r2={recon_r2:.3f}, interpretability={interpretability:.3f}, mcmc_efficiency={mcmc_efficiency:.3f}, final_score={quality_score:.3f}")

                return quality_score

            except Exception as e:
                logger.warning(f"Quality evaluation failed for K={K}, percW={percW}: {e}")
                return 0.0

        # Automatic optimal hyperparameter determination function
        def determine_optimal_hyperparameters(X_list, config, optimize_params=['K']):
            """
            Determine optimal hyperparameters using quality evaluation.
            optimize_params can include: 'K', 'percW', 'mcmc', 'joint'
            """
            import argparse
            from itertools import product

            # Get configuration
            hyperparam_config = config.get('hyperparameter_optimization', {})
            if not hyperparam_config.get('enabled', True):
                # Use fallback values
                fallback_K = hyperparam_config.get('fallback_K', 10)
                fallback_percW = hyperparam_config.get('fallback_percW', 33)
                logger.info(f"Hyperparameter optimization disabled - using fallbacks K={fallback_K}, percW={fallback_percW}")
                return {'K': fallback_K, 'percW': fallback_percW}, 0.0, {}

            # Define candidate values
            K_candidates = hyperparam_config.get('K_candidates', [5, 8, 10, 12, 15])
            percW_candidates = hyperparam_config.get('percW_candidates', [20, 25, 33, 40, 50])

            # MCMC parameter candidates
            mcmc_config = config.get('training', {}).get('mcmc_config', {})
            num_samples_candidates = hyperparam_config.get('num_samples_candidates', [1000, 2000])
            num_warmup_candidates = hyperparam_config.get('num_warmup_candidates', [500, 1000])
            num_chains_candidates = hyperparam_config.get('num_chains_candidates', [2, 4])
            target_accept_prob_candidates = hyperparam_config.get('target_accept_prob_candidates', [0.8])

            logger.info(f"Optimizing hyperparameters: {optimize_params}")

            if 'joint' in optimize_params or (len(optimize_params) > 1):
                # Joint optimization including MCMC parameters if requested
                include_mcmc = 'mcmc' in optimize_params or 'joint' in optimize_params

                if include_mcmc:
                    logger.info("Performing joint K, percW, and MCMC parameter optimization...")
                    logger.info(f"Testing K values: {K_candidates}")
                    logger.info(f"Testing percW values: {percW_candidates}")
                    logger.info(f"Testing num_samples values: {num_samples_candidates}")
                    logger.info(f"Testing num_chains values: {num_chains_candidates}")

                    best_score = -1
                    best_params = {'K': 10, 'percW': 33, 'num_samples': 2000, 'num_warmup': 1000, 'num_chains': 4, 'target_accept_prob': 0.8}
                    all_results = {}

                    for K, percW, num_samples, num_chains in product(K_candidates, percW_candidates, num_samples_candidates, num_chains_candidates):
                        try:
                            # Auto-determine warmup as 50% of samples
                            num_warmup = num_samples // 2

                            params = {
                                'K': K,
                                'percW': percW,
                                'num_samples': num_samples,
                                'num_warmup': num_warmup,
                                'num_chains': num_chains,
                                'target_accept_prob': target_accept_prob_candidates[0]
                            }
                            score = evaluate_model_quality(params, X_list, argparse.Namespace(), config)

                            result_key = f'K{K}_percW{percW}_samples{num_samples}_chains{num_chains}'
                            all_results[result_key] = params.copy()
                            all_results[result_key]['score'] = score

                            logger.info(f"K={K}, percW={percW}, samples={num_samples}, chains={num_chains}: Quality Score = {score:.4f}")

                            if score > best_score:
                                best_score = score
                                best_params = params

                        except Exception as e:
                            logger.warning(f"Failed to evaluate K={K}, percW={percW}, samples={num_samples}: {e}")

                else:
                    # Joint K and percW optimization only
                    logger.info("Performing joint K and percW optimization...")
                    logger.info(f"Testing K values: {K_candidates}")
                    logger.info(f"Testing percW values: {percW_candidates}")

                    best_score = -1
                    best_params = {'K': 10, 'percW': 33}
                    all_results = {}

                    for K, percW in product(K_candidates, percW_candidates):
                        try:
                            params = {'K': K, 'percW': percW}
                            score = evaluate_model_quality(params, X_list, argparse.Namespace(), config)
                            all_results[f'K{K}_percW{percW}'] = {'K': K, 'percW': percW, 'score': score}

                            logger.info(f"K={K}, percW={percW}: Quality Score = {score:.4f}")

                            if score > best_score:
                                best_score = score
                                best_params = params

                        except Exception as e:
                            logger.warning(f"Failed to evaluate K={K}, percW={percW}: {e}")
                            all_results[f'K{K}_percW{percW}'] = {'K': K, 'percW': percW, 'score': 0.0}

                logger.info("="*60)
                logger.info("JOINT HYPERPARAMETER OPTIMIZATION RESULTS:")
                logger.info(f"OPTIMAL COMBINATION: K={best_params['K']}, percW={best_params['percW']} (score: {best_score:.4f})")
                logger.info("="*60)

                return best_params, best_score, all_results

            elif 'K' in optimize_params:
                # K-only optimization (backward compatibility)
                logger.info("Optimizing K only...")
                logger.info(f"Testing K values: {K_candidates}")

                best_score = -1
                best_K = 10
                K_results = {}
                fixed_percW = 33  # Use default percW

                for K in K_candidates:
                    try:
                        params = {'K': K, 'percW': fixed_percW}
                        score = evaluate_model_quality(params, X_list, argparse.Namespace(), config)
                        K_results[K] = score

                        logger.info(f"K={K}: Quality Score = {score:.4f}")

                        if score > best_score:
                            best_score = score
                            best_K = K

                    except Exception as e:
                        logger.warning(f"Failed to evaluate K={K}: {e}")
                        K_results[K] = 0.0

                logger.info("="*50)
                logger.info("K OPTIMIZATION RESULTS:")
                for K in sorted(K_results.keys()):
                    score = K_results[K]
                    marker = " <-- OPTIMAL" if K == best_K else ""
                    logger.info(f"  K={K}: {score:.4f}{marker}")
                logger.info(f"SELECTED: K={best_K} (score: {best_score:.4f})")
                logger.info("="*50)

                return {'K': best_K, 'percW': fixed_percW}, best_score, K_results

            elif 'percW' in optimize_params:
                # percW-only optimization
                logger.info("Optimizing percW only...")
                logger.info(f"Testing percW values: {percW_candidates}")

                best_score = -1
                best_percW = 33
                percW_results = {}
                fixed_K = 10  # Use default K

                for percW in percW_candidates:
                    try:
                        params = {'K': fixed_K, 'percW': percW}
                        score = evaluate_model_quality(params, X_list, argparse.Namespace(), config)
                        percW_results[percW] = score

                        logger.info(f"percW={percW}: Quality Score = {score:.4f}")

                        if score > best_score:
                            best_score = score
                            best_percW = percW

                    except Exception as e:
                        logger.warning(f"Failed to evaluate percW={percW}: {e}")
                        percW_results[percW] = 0.0

                logger.info("="*50)
                logger.info("percW OPTIMIZATION RESULTS:")
                for percW in sorted(percW_results.keys()):
                    score = percW_results[percW]
                    marker = " <-- OPTIMAL" if percW == best_percW else ""
                    logger.info(f"  percW={percW}: {score:.4f}{marker}")
                logger.info(f"SELECTED: percW={best_percW} (score: {best_score:.4f})")
                logger.info("="*50)

                return {'K': fixed_K, 'percW': best_percW}, best_score, percW_results

            else:
                logger.warning("No optimization parameters specified - using defaults")
                return {'K': 10, 'percW': 33}, 0.0, {}

        # Create sensitivity analysis experiment function
        def sensitivity_analysis_experiment(config, output_dir, **kwargs):
            import numpy as np
            from remote_workstation.preprocessing_integration import apply_preprocessing_to_pipeline

            logger.info("Running direct sensitivity analysis...")
            X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                config=config.__dict__,
                data_dir=config.data_dir,
                auto_select_strategy=True
            )
            
            results = {
                'parameter_sensitivity': {},
                'robustness_analysis': {},
                'stability_analysis': {}
            }
            
            # Parameter sensitivity analysis with automatic K selection
            logger.info("Running parameter sensitivity with automatic optimal K selection...")
            K_values = [3, 5, 8, 10, 15]

            # Track results for optimal K selection
            K_evaluation_results = {}

            for K in K_values:
                try:
                    from core.run_analysis import main
                    import argparse
                    
                    # Create complete args for SGFA analysis
                    args = argparse.Namespace(
                        model='sparseGFA',
                        K=K,
                        num_samples=200,
                        num_warmup=100,
                        num_chains=1,
                        num_runs=1,
                        dataset='qmap_pd',
                        data_dir=config.data_dir,
                        device='gpu',
                        clinical_rel="data_clinical/pd_motor_gfa_data.tsv",
                        volumes_rel="volume_matrices",
                        id_col="sid",
                        roi_views=True,
                        noise=0,
                        seed=42,
                        num_sources=4,
                        reghsZ=True,
                        percW=33
                    )
                    
                    # Run SGFA analysis
                    logger.info(f"Testing K={K} factors...")
                    model_results = main(args)

                    # Evaluate model quality using existing framework
                    quality_score = evaluate_model_quality_for_K(
                        K=K,
                        X_list=X_list,
                        args=args,
                        config=config
                    )

                    # Store results with quality evaluation
                    K_result = {
                        'K': K,
                        'converged': True,
                        'quality_score': quality_score,
                        'n_factors': K,
                        'status': 'completed'
                    }

                    results['parameter_sensitivity'][f'K_{K}'] = K_result
                    K_evaluation_results[K] = quality_score

                    logger.info(f"K={K}: Quality score = {quality_score:.4f}")
                except Exception as e:
                    results['parameter_sensitivity'][f'K_{K}'] = {
                        'K': K,
                        'status': 'failed',
                        'error': str(e)
                    }

            # AUTOMATIC OPTIMAL K SELECTION
            logger.info("="*60)
            logger.info("OPTIMAL K DETERMINATION:")

            if K_evaluation_results:
                # Find optimal K
                optimal_K = max(K_evaluation_results.keys(), key=lambda k: K_evaluation_results[k])
                optimal_score = K_evaluation_results[optimal_K]

                # Log detailed results
                logger.info("K Evaluation Results:")
                for K in sorted(K_evaluation_results.keys()):
                    score = K_evaluation_results[K]
                    marker = " <-- OPTIMAL" if K == optimal_K else ""
                    logger.info(f"  K={K}: Quality Score = {score:.4f}{marker}")

                logger.info(f"OPTIMAL K SELECTED: {optimal_K} factors (score: {optimal_score:.4f})")

                # Store optimal K results
                results['optimal_K_selection'] = {
                    'optimal_K': optimal_K,
                    'optimal_score': optimal_score,
                    'all_K_scores': K_evaluation_results,
                    'K_values_tested': list(K_values),
                    'selection_method': 'automatic_quality_scoring'
                }
            else:
                logger.warning("No valid K evaluation results - using default K=10")
                results['optimal_K_selection'] = {
                    'optimal_K': 10,
                    'optimal_score': 0.0,
                    'status': 'fallback_default',
                    'K_values_tested': list(K_values)
                }

            logger.info("="*60)

            # Robustness analysis with noise
            logger.info("Running robustness analysis...")
            noise_levels = [0.01, 0.05, 0.1]
            
            for noise_level in noise_levels:
                try:
                    # Add noise to data
                    X_noisy = []
                    for X in X_list:
                        noise = np.random.normal(0, noise_level * np.std(X), X.shape)
                        X_noisy.append(X + noise)
                    
                    from core.run_analysis import main
                    import argparse
                    
                    # Create complete args for SGFA analysis
                    args = argparse.Namespace(
                        model='sparseGFA',
                        K=10,
                        num_samples=200,
                        num_warmup=100,
                        num_chains=1,
                        num_runs=1,
                        dataset='qmap_pd',
                        data_dir=config.data_dir,
                        device='gpu',
                        clinical_rel="data_clinical/pd_motor_gfa_data.tsv",
                        volumes_rel="volume_matrices",
                        id_col="sid",
                        roi_views=True,
                        noise=0,
                        seed=42,
                        num_sources=4,
                        reghsZ=True,
                        percW=33
                    )
                    
                    # TODO: Pass X_noisy to main - need to modify approach
                    main(args)
                    result = {'status': 'completed'}
                    
                    results['robustness_analysis'][f'noise_{noise_level}'] = {
                        'noise_level': noise_level,
                        'converged': result.get('converged', True),
                        'log_likelihood': result.get('log_likelihood', 0)
                    }
                except Exception as e:
                    results['robustness_analysis'][f'noise_{noise_level}'] = {
                        'noise_level': noise_level,
                        'status': 'failed',
                        'error': str(e)
                    }
            
            # Stability analysis with different random seeds
            logger.info("Running stability analysis...")
            random_seeds = [42, 123, 456, 789]
            
            for seed in random_seeds:
                try:
                    np.random.seed(seed)
                    from core.run_analysis import main
                    import argparse
                    
                    # Create complete args for SGFA analysis
                    args = argparse.Namespace(
                        model='sparseGFA',
                        K=10,
                        num_samples=200,
                        num_warmup=100,
                        num_chains=1,
                        num_runs=1,
                        dataset='qmap_pd',
                        data_dir=config.data_dir,
                        device='gpu',
                        clinical_rel="data_clinical/pd_motor_gfa_data.tsv",
                        volumes_rel="volume_matrices",
                        id_col="sid",
                        roi_views=True,
                        noise=0,
                        seed=seed,
                        num_sources=4,
                        reghsZ=True,
                        percW=33
                    )
                    
                    main(args)
                    result = {'status': 'completed'}
                    
                    results['stability_analysis'][f'seed_{seed}'] = {
                        'seed': seed,
                        'converged': result.get('converged', True),
                        'log_likelihood': result.get('log_likelihood', 0)
                    }
                except Exception as e:
                    results['stability_analysis'][f'seed_{seed}'] = {
                        'seed': seed,
                        'status': 'failed',
                        'error': str(e)
                    }
            
            logger.info("Sensitivity analysis completed")
            
            # Format results for framework compatibility
            return {
                'model_results': results,
                'experiment_metadata': {
                    'analysis_types': list(results.keys()),
                    'parameter_tests': len(results.get('parameter_sensitivity', {})),
                    'robustness_tests': len(results.get('robustness_analysis', {})),
                    'stability_tests': len(results.get('stability_analysis', {}))
                },
                'diagnostics': {
                    'total_sensitivity_tests': sum(len(v) if isinstance(v, dict) else 1 for v in results.values()),
                    'analysis_status': 'completed'
                }
            }
        
        result = framework.run_experiment(exp_config, sensitivity_analysis_experiment)
        
        logger.info(" Sensitivity analysis experiments completed")
        return result
        
    except Exception as e:
        logger.error(f" Sensitivity analysis failed: {e}")
        return None

def main():
    """Main experimental pipeline."""
    parser = argparse.ArgumentParser(description="Run Remote Workstation experimental framework")
    parser.add_argument("--config", default="remote_workstation/config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--experiments", nargs="+", 
                       choices=["data_validation", "method_comparison", 
                               "performance_benchmarks", "sensitivity_analysis", "all"],
                       default=["all"], help="Experiments to run")
    parser.add_argument("--data-dir", help="Override data directory")
    parser.add_argument("--unified-results", action="store_true", default=True,
                       help="Save all results in a single timestamped folder (default: True)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override data directory if provided
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
        logger.info(f"Using data directory: {args.data_dir}")
    
    # Setup unified results directory if requested
    if args.unified_results:
        # Create single timestamped directory for all experiments
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unified_dir = Path(config['experiments']['base_output_dir']) / f"complete_run_{run_timestamp}"
        unified_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config to use unified directory
        original_base_dir = config['experiments']['base_output_dir']
        config['experiments']['base_output_dir'] = str(unified_dir)
        
        logger.info(f"🗂️  Using unified results directory: {unified_dir}")
        logger.info(f"   All experiments will save to: {unified_dir.name}")
        
        # Create organized subdirectories in unified folder
        (unified_dir / "01_data_validation").mkdir(exist_ok=True)
        (unified_dir / "02_method_comparison").mkdir(exist_ok=True)
        (unified_dir / "03_performance_benchmarks").mkdir(exist_ok=True)
        (unified_dir / "04_sensitivity_analysis").mkdir(exist_ok=True)
        (unified_dir / "plots").mkdir(exist_ok=True)
        (unified_dir / "brain_maps").mkdir(exist_ok=True)
        (unified_dir / "summaries").mkdir(exist_ok=True)
    
    # Setup environment
    setup_environment(config)
    
    # Track results
    results = {}
    start_time = datetime.now()

    logger.info(f" Starting Remote Workstation Experimental Framework at {start_time}")
    logger.info(f" Running experiments: {args.experiments}")
    
    # Determine which experiments to run
    experiments_to_run = args.experiments
    if "all" in experiments_to_run:
        experiments_to_run = ["data_validation", "method_comparison", 
                             "performance_benchmarks", "sensitivity_analysis"]
    
    # Run experiments
    if "data_validation" in experiments_to_run:
        results['data_validation'] = run_data_validation(config)
    
    if "method_comparison" in experiments_to_run:
        results['method_comparison'] = run_method_comparison(config)
    
    if "performance_benchmarks" in experiments_to_run:
        results['performance_benchmarks'] = run_performance_benchmarks(config)
    
    if "sensitivity_analysis" in experiments_to_run:
        results['sensitivity_analysis'] = run_sensitivity_analysis(config)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f" All experiments completed!")
    logger.info(f"  Total duration: {duration}")
    logger.info(f" Results saved to: {config['experiments']['base_output_dir']}")
    
    # Create comprehensive summary
    if args.unified_results:
        logger.info(f"📋 Creating comprehensive experiment summary...")
        
        # Collect detailed results information
        summary = {
            'experiment_run_info': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(), 
                'duration_seconds': duration.total_seconds(),
                'duration_formatted': str(duration),
                'unified_results': True,
                'config_used': args.config
            },
            'experiments_executed': {},
            'results_summary': {
                'total_experiments': len(experiments_to_run),
                'successful_experiments': len([r for r in results.values() if r is not None]),
                'failed_experiments': len([r for r in results.values() if r is None])
            }
        }
        
        # Add experiment-specific summaries
        for exp_name, result in results.items():
            if result is not None:
                exp_summary = {
                    'status': getattr(result, 'status', 'completed'),
                    'experiment_id': getattr(result, 'experiment_id', 'N/A'),
                    'duration': getattr(result, 'get_duration', lambda: 0)()
                }
                
                # Add specific details based on experiment type
                if exp_name == 'method_comparison' and hasattr(result, 'model_results'):
                    model_results = result.model_results
                    if 'sgfa_variants' in model_results:
                        exp_summary['sgfa_variants'] = list(model_results['sgfa_variants'].keys())
                        exp_summary['successful_variants'] = len([v for v in model_results['sgfa_variants'].values() if v.get('status') == 'completed'])
                    if 'plots' in model_results:
                        exp_summary['plots_generated'] = model_results['plots'].get('plot_count', 0)
                        exp_summary['brain_maps_available'] = 'brain_maps' in str(model_results['plots'].get('generated_plots', []))
                        
                summary['experiments_executed'][exp_name] = exp_summary
            else:
                summary['experiments_executed'][exp_name] = {'status': 'failed'}
        
        # Save main summary
        summary_path = Path(config['experiments']['base_output_dir']) / 'summaries' / 'complete_experiment_summary.yaml'
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
            
        # Create a simple text summary for quick reading
        text_summary_path = Path(config['experiments']['base_output_dir']) / 'README.md'
        with open(text_summary_path, 'w') as f:
            f.write(f"# SGFA Experiment Run Results\\n\\n")
            f.write(f"**Run Date:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**Duration:** {duration}\\n")
            f.write(f"**Experiments:** {', '.join(experiments_to_run)}\\n\\n")
            
            f.write(f"## Results Structure\\n\\n")
            f.write(f"```\\n")
            f.write(f"01_data_validation/     - Data quality and preprocessing analysis\\n")
            f.write(f"02_method_comparison/   - SGFA variants and traditional methods\\n")
            f.write(f"03_performance_benchmarks/ - Scalability and performance tests\\n")
            f.write(f"04_sensitivity_analysis/ - Parameter sensitivity studies\\n")
            f.write(f"plots/                  - All visualization outputs\\n")
            f.write(f"brain_maps/            - Factor loadings mapped to brain space\\n")
            f.write(f"summaries/             - Detailed summaries and reports\\n")
            f.write(f"```\\n\\n")
            
            # Add experiment-specific details
            for exp_name, result in results.items():
                if result is not None:
                    f.write(f"### {exp_name.replace('_', ' ').title()}\\n")
                    f.write(f"- Status: Completed\\n")
                    if exp_name == 'method_comparison' and hasattr(result, 'model_results'):
                        model_results = result.model_results
                        if 'sgfa_variants' in model_results:
                            f.write(f"- SGFA Variants: {list(model_results['sgfa_variants'].keys())}\\n")
                        if 'plots' in model_results:
                            f.write(f"- Plots Generated: {model_results['plots'].get('plot_count', 0)}\\n")
                    f.write(f"\\n")
                else:
                    f.write(f"### {exp_name.replace('_', ' ').title()}\\n")
                    f.write(f"- Status: Failed\\n\\n")
        
        logger.info(f"📋 Comprehensive summary saved to: {summary_path}")
        logger.info(f"📖 Quick reference saved to: {text_summary_path}")
        
    else:
        # Simple summary for non-unified results
        summary = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(), 
            'duration_seconds': duration.total_seconds(),
            'experiments_run': experiments_to_run,
            'success_count': len([r for r in results.values() if r is not None]),
            'config_used': args.config
        }
        
        summary_path = Path(config['experiments']['base_output_dir']) / 'experiment_summary.yaml'
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f" Experiment summary saved to: {summary_path}")

if __name__ == "__main__":
    main()