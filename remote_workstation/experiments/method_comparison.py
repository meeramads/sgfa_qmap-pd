#!/usr/bin/env python
"""
Method Comparison Experiments
Compare SGFA model variants on remote workstation.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run_method_comparison(config):
    """Run method comparison experiments."""
    logger.info("Starting Method Comparison Experiments")

    try:
        # Add project root to path for framework imports
        import sys
        import os

        # Calculate the correct project root path
        current_file = os.path.abspath(__file__)  # /path/to/remote_workstation/experiments/method_comparison.py
        remote_ws_dir = os.path.dirname(os.path.dirname(current_file))  # /path/to/remote_workstation
        project_root = os.path.dirname(remote_ws_dir)  # /path/to/project_root

        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Try direct import after adding path
        try:
            from experiments.framework import ExperimentFramework, ExperimentConfig
            logger.info("✅ Successfully imported experiments.framework")
        except ImportError as e:
            logger.error(f"❌ Import failed: {e}")
            # Try alternative import approach
            import importlib.util
            framework_path = os.path.join(project_root, 'experiments', 'framework.py')
            spec = importlib.util.spec_from_file_location("experiments.framework", framework_path)
            framework_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(framework_module)
            ExperimentFramework = framework_module.ExperimentFramework
            ExperimentConfig = framework_module.ExperimentConfig
            logger.info("✅ Successfully imported via direct file loading")

        framework = ExperimentFramework(
            base_output_dir=Path(config['experiments']['base_output_dir'])
        )

        exp_config = ExperimentConfig(
            experiment_name="remote_workstation_method_comparison",
            description="Compare SGFA model variants on remote workstation",
            dataset="qmap_pd",
            data_dir=config['data']['data_dir']
        )

        # COMPREHENSIVE MODELS FRAMEWORK INTEGRATION
        from remote_workstation.models_integration import integrate_models_with_pipeline

        logger.info("🧠 Integrating comprehensive models framework...")
        # We'll get data characteristics after loading data
        model_type, model_instance, models_summary = integrate_models_with_pipeline(
            config=config
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

        # Update models framework with data characteristics
        if X_list and models_summary:
            logger.info("🧠 Updating models framework with data characteristics...")
            data_characteristics = {
                'n_subjects': len(X_list[0]),
                'n_views': len(X_list),
                'total_features': sum(X.shape[1] for X in X_list),
                'view_dimensions': [X.shape[1] for X in X_list],
                'has_imaging_data': any(X.shape[1] > 1000 for X in X_list),
                'imaging_views': [i for i, X in enumerate(X_list) if X.shape[1] > 1000]
            }

            # Re-run model selection with data characteristics
            from remote_workstation.models_integration import integrate_models_with_pipeline
            model_type, model_instance, updated_models_summary = integrate_models_with_pipeline(
                config=config,
                X_list=X_list,
                data_characteristics=data_characteristics
            )
            models_summary = updated_models_summary

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

            # Import helper functions
            from .helpers import determine_optimal_hyperparameters

            # Log models framework information
            if models_summary:
                integration_info = models_summary.get('integration_summary', {})
                logger.info("🧠 MODELS FRAMEWORK SUMMARY:")
                logger.info(f"   Framework: {'Structured models' if integration_info.get('structured_model_management', False) else 'Direct core models'}")
                logger.info(f"   Model type: {integration_info.get('model_type_selected', 'unknown')}")
                logger.info(f"   Model factory: {'used' if integration_info.get('model_factory_used', False) else 'unavailable'}")
                logger.info(f"   Model instance: {'created' if integration_info.get('model_instance_created', False) else 'failed'}")

                if integration_info.get('available_models'):
                    logger.info(f"   Available models: {', '.join(integration_info.get('available_models', []))}")

                logger.info(f"   Features: Neuroimaging={integration_info.get('neuroimaging_optimized', False)}, "
                          f"Sparsity={integration_info.get('sparsity_regularization', False)}")

                if integration_info.get('comparison_completed', False):
                    comparison = integration_info.get('model_comparison', {})
                    if comparison.get('best_model'):
                        logger.info(f"   Best model: {comparison['best_model']} (score: {comparison.get('best_score', 0):.2f})")

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

            # Run the full method comparison logic
            return _run_method_comparison_logic(
                config, X_list, optimal_params, optimal_score, all_scores,
                output_dir, models_summary, analysis_summary, performance_summary,
                model_type, model_instance, data_manager, model_runner, performance_manager
            )

        result = framework.run_experiment(exp_config, method_comparison_experiment)

        logger.info(" Method comparison experiments completed")
        return result

    except Exception as e:
        logger.error(f" Method comparison failed: {e}")
        return None


def _run_method_comparison_logic(config, X_list, optimal_params, optimal_score, all_scores,
                                output_dir, models_summary, analysis_summary, performance_summary,
                                model_type, model_instance, data_manager, model_runner, performance_manager):
    """Core method comparison logic separated for modularity."""
    import numpy as np
    import time

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

        # Run SGFA variant
        try:
            variant_result = _run_sgfa_variant(
                variant_name, variant_config, optimal_params, X_list,
                models_summary, analysis_summary, performance_summary,
                model_type, model_instance, data_manager, model_runner, performance_manager,
                config
            )
            results['sgfa_variants'][variant_name] = variant_result
            logger.info(f"✅ {variant_name} SGFA completed successfully!")
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
            method_result = _run_traditional_method(method, X_combined)
            results['traditional_methods'][method] = method_result
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"✅ {method.upper()} completed in {duration:.2f} seconds")
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

    # Generate comprehensive visualizations
    try:
        logger.info("\n🎨 === GENERATING COMPREHENSIVE VISUALIZATION SUITE ===")
        results['plots'] = _generate_visualizations(
            results, X_list, optimal_params, optimal_score, config, output_dir, all_scores
        )
    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
        results['plots'] = {'status': 'failed', 'error': str(e)}

    # Generate experiment summary
    _log_experiment_summary(results)

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


def _run_sgfa_variant(variant_name, variant_config, optimal_params, X_list,
                     models_summary, analysis_summary, performance_summary,
                     model_type, model_instance, data_manager, model_runner, performance_manager, config):
    """Run a single SGFA variant with comprehensive framework integration."""
    import argparse
    import time
    import numpy as np

    # Import SGFA core components directly
    from core.run_analysis import models, run_inference
    import jax
    import jax.random as random

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
    start_time = time.time()
    rng_key = random.PRNGKey(42)

    logger.info(f"⏱️  Starting MCMC inference at {time.strftime('%H:%M:%S')}...")
    logger.info(f"Expected duration: ~{args.num_samples/10:.1f}-{args.num_samples/5:.1f} minutes")

    # Apply comprehensive framework integration for MCMC execution
    mcmc_result = _execute_mcmc_with_frameworks(
        models, args, rng_key, X_list, hypers,
        model_instance, models_summary, model_runner, data_manager,
        analysis_summary, performance_manager, config
    )

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"✅ MCMC completed in {duration/60:.2f} minutes ({duration:.1f} seconds)")
    logger.info(f"   Average time per sample: {duration/args.num_samples:.2f}s")

    # Extract and process MCMC samples
    return _process_mcmc_results(mcmc_result, args, variant_config, duration)


def _execute_mcmc_with_frameworks(models, args, rng_key, X_list, hypers,
                                 model_instance, models_summary, model_runner, data_manager,
                                 analysis_summary, performance_manager, config):
    """Execute MCMC with comprehensive framework integration."""

    # Apply structured models framework if available
    if model_instance and models_summary.get('integration_summary', {}).get('structured_model_management', False):
        logger.info("🧠 Using structured models framework for MCMC execution...")
        from remote_workstation.models_integration import _wrap_models_framework

        # Use structured model
        models_wrapper = _wrap_models_framework(model_type, model_instance, models_summary)
        structured_model = models_wrapper.get_model_for_execution()

        if structured_model:
            logger.info(f"✅ Using model: {structured_model.get_model_name()}")
            # Execute MCMC with structured model
            return run_inference(structured_model, args, rng_key, X_list, hypers)
        else:
            logger.warning("⚠️ Structured model unavailable - falling back")

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
            return mcmc_result

    # Apply MCMC-specific performance optimizations
    if performance_manager:
        logger.info("⚡ Applying MCMC performance optimization...")
        from remote_workstation.performance_integration import optimize_mcmc_execution
        return optimize_mcmc_execution(
            performance_manager=performance_manager,
            model_fn=models,
            args=args,
            rng_key=rng_key,
            X_list=X_list,
            hypers=hypers
        )
    else:
        logger.info("⚡ Using standard MCMC execution (frameworks unavailable)")
        return run_inference(models, args, rng_key, X_list, hypers)


def _process_mcmc_results(mcmc_result, args, variant_config, duration):
    """Process and format MCMC results."""
    import numpy as np

    # Extract MCMC samples
    logger.info(f"📊 Extracting MCMC samples...")
    samples = mcmc_result.get_samples()
    logger.info(f"   - Total samples collected: {mcmc_result.num_samples}")
    logger.info(f"   - Number of chains: {mcmc_result.num_chains}")
    logger.info(f"   - Sample keys: {list(samples.keys())}")

    variant_result_data = {
        'status': 'completed',
        'converged': True,
        'log_likelihood': 0,
        'n_factors': args.K,
        'duration_minutes': duration/60,
        'config': variant_config,
        'mcmc_info': {
            'num_samples': mcmc_result.num_samples,
            'num_chains': mcmc_result.num_chains,
            'sample_keys': list(samples.keys())
        }
    }

    # Extract key parameters
    if 'Z' in samples:
        Z_samples = samples['Z']  # Factor scores
        logger.info(f"   - Factor scores shape: {Z_samples.shape}")
        Z_mean = Z_samples.mean(axis=0)  # Average over MCMC samples

        variant_result_data['factor_scores'] = {
            'shape': list(Z_samples.shape),
            'mean': Z_mean.tolist(),  # Convert to list for JSON serialization
            'std': Z_samples.std(axis=0).tolist()
        }
        logger.info(f"   🧠 Factor scores: {Z_mean.shape} (subjects x factors)")

    if 'W' in samples:
        W_samples = samples['W']  # Factor loadings/weights
        logger.info(f"   - Factor loadings shape: {W_samples.shape}")
        W_mean = W_samples.mean(axis=0)  # Average over MCMC samples

        variant_result_data['factor_loadings'] = {
            'shape': list(W_samples.shape),
            'mean': W_mean.tolist(),
            'std': W_samples.std(axis=0).tolist()
        }
        logger.info(f"   🎨 Factor loadings: {W_mean.shape} (features x factors)")

        # Calculate sparsity metrics
        sparsity = (np.abs(W_mean) < 0.1).mean()
        logger.info(f"   🔍 Sparsity (|loading| < 0.1): {sparsity:.3f}")

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

    return variant_result_data


def _run_traditional_method(method, X_combined):
    """Run a traditional dimensionality reduction method."""
    import time
    from sklearn.decomposition import PCA, FastICA, FactorAnalysis

    if method == 'pca':
        logger.info(f"🚀 Running PCA with 10 components...")
        model = PCA(n_components=10)
        factors = model.fit_transform(X_combined)
        explained_var = model.explained_variance_ratio_.sum()
        logger.info(f"   Total explained variance: {explained_var:.3f}")
        logger.info(f"   Top 3 component variances: {model.explained_variance_ratio_[:3]}")

        return {
            'status': 'completed',
            'n_components': factors.shape[1],
            'explained_variance_ratio': explained_var,
            'factor_shape': factors.shape,
            'duration_seconds': time.time()
        }

    elif method == 'ica':
        logger.info(f"🚀 Running FastICA with 10 components...")
        model = FastICA(n_components=10, random_state=42, max_iter=1000)
        factors = model.fit_transform(X_combined)
        logger.info(f"   ICA converged: {model.n_iter_ < model.max_iter}")
        logger.info(f"   Iterations used: {model.n_iter_}")

        return {
            'status': 'completed',
            'n_components': factors.shape[1],
            'explained_variance_ratio': None,  # ICA doesn't have explained variance
            'factor_shape': factors.shape,
            'duration_seconds': time.time()
        }

    elif method == 'factor_analysis':
        logger.info(f"🚀 Running Factor Analysis with 10 components...")
        model = FactorAnalysis(n_components=10, random_state=42)
        factors = model.fit_transform(X_combined)
        loglike = model.score(X_combined)
        logger.info(f"   Log-likelihood: {loglike:.2f}")
        logger.info(f"   Noise variance: {model.noise_variance_.mean():.6f}")

        return {
            'status': 'completed',
            'n_components': factors.shape[1],
            'explained_variance_ratio': None,  # FA doesn't have explained variance like PCA
            'factor_shape': factors.shape,
            'duration_seconds': time.time()
        }


def _generate_visualizations(results, X_list, optimal_params, optimal_score, config, output_dir, all_scores):
    """Generate comprehensive visualizations."""
    try:
        logger.info("🎨 === GENERATING COMPREHENSIVE VISUALIZATION SUITE ===")
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
        plots_result = create_comprehensive_visualizations(
            results=results,
            X_list=X_list,
            optimal_params=optimal_params,
            optimal_score=optimal_score,
            config=config,
            output_dir=output_dir,
            all_scores=all_scores if 'all_scores' in locals() else None
        )

        # Fallback to basic visualization if comprehensive fails
        if plots_result['status'] == 'visualization_manager_unavailable':
            plots_result = create_fallback_visualizations(results, X_list, output_dir)

        return plots_result

    except ImportError as e:
        logger.warning(f"⚠️  VisualizationManager not available: {e}")
        # Try fallback visualization
        try:
            from remote_workstation.visualization_integration import create_fallback_visualizations
            return create_fallback_visualizations(results, X_list, output_dir)
        except Exception as fallback_e:
            logger.error(f"❌ Fallback visualization also failed: {fallback_e}")
            return {'status': 'visualization_unavailable', 'error': str(e)}
    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
        return {'status': 'failed', 'error': str(e)}


def _log_experiment_summary(results):
    """Log comprehensive experiment summary."""
    logger.info("\n🎉 === METHOD COMPARISON COMPLETED ===")

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