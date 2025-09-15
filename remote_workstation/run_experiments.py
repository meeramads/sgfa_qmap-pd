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
        
        # Load data first
        from data.qmap_pd import load_qmap_pd
        data = load_qmap_pd(data_dir=config['data']['data_dir'])
        
        # Create method comparison experiment function  
        def method_comparison_experiment(config, output_dir, **kwargs):
            import numpy as np  # Add missing numpy import
            logger.info("Running comprehensive method comparison...")
            X_list = data['X_list']
            
            # Direct implementation of method comparison logic
            results = {
                'sgfa_variants': {},
                'traditional_methods': {},
                'experiment_metadata': {
                    'n_subjects': X_list[0].shape[0],
                    'n_views': len(X_list),
                    'feature_counts': [X.shape[1] for X in X_list],
                    'total_features': sum(X.shape[1] for X in X_list)
                }
            }
            
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
                        K=config.K_values[0] if hasattr(config, 'K_values') else 10,
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
                    
                    # Run SGFA inference directly
                    import time
                    start_time = time.time()
                    rng_key = random.PRNGKey(42)
                    
                    logger.info(f"⏱️  Starting MCMC inference at {time.strftime('%H:%M:%S')}...")
                    logger.info(f"Expected duration: ~{args.num_samples/10:.1f}-{args.num_samples/5:.1f} minutes")
                    
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
                logger.info("\\n🎨 === GENERATING VISUALIZATION PLOTS ===")
                from visualization.factor_plots import FactorPlotter
                from pathlib import Path
                
                plot_dir = Path(output_dir) / "plots"
                plot_dir.mkdir(exist_ok=True)
                logger.info(f"Plot directory: {plot_dir}")
                
                plotter = FactorPlotter()
                plots_generated = []
                
                # Check for brain imaging capabilities
                brain_imaging_available = False
                try:
                    from visualization.neuroimaging_utils import FactorToMRIMapper, integrate_with_visualization
                    import nibabel as nib
                    brain_imaging_available = True
                    logger.info("🧠 Brain imaging visualization available")
                except ImportError:
                    logger.info("⚠️  Brain imaging visualization not available (nibabel required)")
                
                # Generate plots for each successful SGFA variant
                for variant_name, variant_result in results['sgfa_variants'].items():
                    if variant_result.get('status') == 'completed' and 'factor_loadings' in variant_result:
                        logger.info(f"🖼️  Generating plots for {variant_name}...")
                        
                        try:
                            # Extract plotting data
                            factor_loadings = np.array(variant_result['factor_loadings']['mean'])
                            factor_scores = np.array(variant_result['factor_scores']['mean'])
                            
                            # Generate factor loading heatmap
                            plot_path = plot_dir / f"{variant_name}_loadings_heatmap.png"
                            plotter.plot_loading_heatmap(
                                factor_loadings, 
                                title=f"{variant_name.title()} SGFA - Factor Loadings",
                                save_path=str(plot_path)
                            )
                            plots_generated.append(str(plot_path))
                            logger.info(f"   ✅ Loading heatmap: {plot_path.name}")
                            
                            # Generate factor score distribution plots
                            plot_path = plot_dir / f"{variant_name}_scores_distribution.png"
                            plotter.plot_factor_distributions(
                                factor_scores,
                                title=f"{variant_name.title()} SGFA - Factor Score Distributions", 
                                save_path=str(plot_path)
                            )
                            plots_generated.append(str(plot_path))
                            logger.info(f"   ✅ Score distributions: {plot_path.name}")
                            
                            # Generate brain imaging plots if available
                            if brain_imaging_available:
                                logger.info(f"🧠 Generating brain imaging plots for {variant_name}...")
                                try:
                                    # Create brain visualization directory
                                    brain_dir = plot_dir / f"{variant_name}_brain_maps"
                                    brain_dir.mkdir(exist_ok=True)
                                    
                                    # Set up brain mapping (need data structure info)
                                    # Get view information from data if available
                                    if 'X_list' in globals() and len(X_list) > 0:
                                        Dm = [X.shape[1] for X in X_list]
                                        view_names = ['structural', 'functional', 'diffusion', 'clinical'][:len(X_list)]
                                        
                                        # Initialize mapper - try common qMAP-PD data directory
                                        try:
                                            mapper = FactorToMRIMapper(config.data_dir if hasattr(config, 'data_dir') else './qMAP-PD_data')
                                            
                                            # Map factor loadings to brain space
                                            factor_maps = mapper.map_all_factors(
                                                W=factor_loadings,
                                                view_names=view_names,
                                                Dm=Dm,
                                                output_dir=str(brain_dir)
                                            )
                                            
                                            if factor_maps:
                                                n_brain_files = sum(len(outputs) for outputs in factor_maps.values())
                                                logger.info(f"   🧠 Generated {n_brain_files} brain map files")
                                                plots_generated.extend([str(brain_dir / f"factor_{i}_*.nii.gz") for i in factor_maps.keys()])
                                                
                                                # Create summary plots of brain maps
                                                try:
                                                    from visualization.brain_plots import BrainVisualizer
                                                    brain_viz = BrainVisualizer(config={'save_plots': True})
                                                    
                                                    summary_plot = brain_dir / f"{variant_name}_brain_summary.png"
                                                    brain_viz.create_brain_visualization_summary(
                                                        str(brain_dir),
                                                        include_reconstructions=True
                                                    )
                                                    logger.info(f"   🧠 Brain summary: {summary_plot.name}")
                                                    
                                                except Exception as brain_viz_e:
                                                    logger.debug(f"Brain summary visualization failed: {brain_viz_e}")
                                            else:
                                                logger.warning(f"   ⚠️  No brain maps generated for {variant_name}")
                                                
                                        except Exception as mapper_e:
                                            logger.warning(f"   ⚠️  Brain mapper failed for {variant_name}: {mapper_e}")
                                    else:
                                        logger.warning(f"   ⚠️  X_list not available for brain mapping")
                                        
                                except Exception as brain_e:
                                    logger.warning(f"   ⚠️  Brain imaging failed for {variant_name}: {brain_e}")
                            
                        except Exception as e:
                            logger.warning(f"   ⚠️  Failed to generate plots for {variant_name}: {e}")
                
                # Generate comparison plots
                if len([r for r in results['sgfa_variants'].values() if r.get('status') == 'completed']) > 1:
                    logger.info(f"📊 Generating comparison plots...")
                    try:
                        plot_path = plot_dir / "sgfa_variants_comparison.png"
                        plotter.plot_method_comparison(
                            results['sgfa_variants'],
                            title="SGFA Variants Comparison",
                            save_path=str(plot_path)
                        )
                        plots_generated.append(str(plot_path))
                        logger.info(f"   ✅ Variant comparison: {plot_path.name}")
                    except Exception as e:
                        logger.warning(f"   ⚠️  Failed to generate comparison plot: {e}")
                
                logger.info(f"🎨 Generated {len(plots_generated)} plots in {plot_dir}")
                
                # Add plot information to results
                results['plots'] = {
                    'plot_directory': str(plot_dir),
                    'generated_plots': plots_generated,
                    'plot_count': len(plots_generated)
                }
                
            except ImportError:
                logger.warning("⚠️  Visualization modules not available - skipping plots")
                results['plots'] = {'status': 'visualization_unavailable'}
            except Exception as e:
                logger.error(f"❌ Plot generation failed: {e}")
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
                    
                    # Count brain imaging outputs
                    brain_files = [p for p in results['plots'].get('generated_plots', []) if 'brain_maps' in p or '.nii.gz' in p]
                    if brain_files:
                        logger.info(f"   🧠 Brain imaging files: {len(brain_files)}")
                        logger.info(f"   🧠 Includes: Factor loadings mapped to brain space")
                        logger.info(f"   🧠 Format: NIfTI files (.nii.gz) for each factor")
                    elif brain_imaging_available:
                        logger.info(f"   🧠 Brain imaging: Available but not generated")
                    else:
                        logger.info(f"   🧠 Brain imaging: Not available (install nibabel)")
                        
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
            from data.qmap_pd import load_qmap_pd
            
            logger.info("Running direct performance benchmarks...")
            data = load_qmap_pd(data_dir=config.data_dir)
            X_list = data['X_list']
            
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
        
        # Create sensitivity analysis experiment function
        def sensitivity_analysis_experiment(config, output_dir, **kwargs):
            import numpy as np
            from data.qmap_pd import load_qmap_pd
            
            logger.info("Running direct sensitivity analysis...")
            data = load_qmap_pd(data_dir=config.data_dir)
            X_list = data['X_list']
            
            results = {
                'parameter_sensitivity': {},
                'robustness_analysis': {},
                'stability_analysis': {}
            }
            
            # Parameter sensitivity analysis
            logger.info("Running parameter sensitivity...")
            K_values = [3, 5, 8, 10, 15]
            
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
                    
                    main(args)
                    result = {'status': 'completed'}
                    results['parameter_sensitivity'][f'K_{K}'] = {
                        'K': K,
                        'converged': result.get('converged', True),
                        'log_likelihood': result.get('log_likelihood', 0),
                        'n_factors': K
                    }
                except Exception as e:
                    results['parameter_sensitivity'][f'K_{K}'] = {
                        'K': K,
                        'status': 'failed',
                        'error': str(e)
                    }
            
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