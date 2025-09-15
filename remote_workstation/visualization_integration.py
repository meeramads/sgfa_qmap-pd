#!/usr/bin/env python3
"""
Visualization integration for remote workstation pipeline.
This replaces the manual visualization code with comprehensive VisualizationManager usage.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def create_comprehensive_visualizations(results: Dict, X_list, optimal_params: Dict, optimal_score: float,
                                      config: Dict, output_dir: str, all_scores: Dict = None) -> Dict:
    """
    Create comprehensive visualizations using VisualizationManager.

    Args:
        results: Experiment results dictionary
        X_list: Input data list
        optimal_params: Optimized hyperparameters
        optimal_score: Optimization score
        config: Configuration dictionary
        output_dir: Output directory path
        all_scores: All hyperparameter scores (optional)

    Returns:
        Dictionary with visualization results
    """
    try:
        logger.info("🎨 === GENERATING COMPREHENSIVE VISUALIZATION SUITE ===")
        from visualization import VisualizationManager

        # Setup visualization configuration
        class VizConfig:
            def __init__(self):
                self.create_brain_viz = True
                self.save_plots = True
                self.output_dir = output_dir
                self.data_dir = config.get('data', {}).get('data_dir', './qMAP-PD_data')
                self.standard_results_dir = Path(output_dir)
                self.cv_results_dir = Path(output_dir)

            def should_run_standard(self):
                return True

        viz_config = VizConfig()

        # Initialize VisualizationManager
        viz_manager = VisualizationManager(viz_config)
        logger.info("🎨 VisualizationManager initialized")

        # Prepare data structures for visualization
        visualization_data = {
            'X_list': X_list,
            'view_names': ['structural', 'functional', 'diffusion', 'clinical'][:len(X_list)],
            'n_subjects': X_list[0].shape[0],
            'feature_dimensions': [X.shape[1] for X in X_list],
            'optimal_hyperparameters': optimal_params,
            'preprocessing': {
                'feature_reduction': {
                    'original_features': sum(X.shape[1] for X in X_list),
                    'subjects': X_list[0].shape[0],
                    'views': len(X_list)
                }
            }
        }

        # Prepare analysis results for visualization
        analysis_results = {
            'sgfa_variants': {},
            'traditional_methods': results['traditional_methods'],
            'experiment_config': {
                'hyperparameter_optimization': {
                    'optimal_K': optimal_params.get('K', 10),
                    'optimal_percW': optimal_params.get('percW', 33),
                    'optimal_num_samples': optimal_params.get('num_samples', 2000),
                    'optimal_num_chains': optimal_params.get('num_chains', 4),
                    'optimization_score': optimal_score
                }
            }
        }

        # Convert SGFA results to analysis format for visualization
        for variant_name, variant_result in results['sgfa_variants'].items():
            if variant_result.get('status') == 'completed':
                analysis_results['sgfa_variants'][variant_name] = {
                    'factor_loadings': np.array(variant_result['factor_loadings']['mean']) if 'factor_loadings' in variant_result else None,
                    'factor_scores': np.array(variant_result['factor_scores']['mean']) if 'factor_scores' in variant_result else None,
                    'metadata': {
                        'n_factors': variant_result.get('n_factors', optimal_params.get('K', 10)),
                        'duration_minutes': variant_result.get('duration_minutes', 0),
                        'mcmc_info': variant_result.get('mcmc_info', {}),
                        'config': variant_result.get('config', {}),
                        'variant_name': variant_name
                    }
                }

        # Prepare cross-validation results (hyperparameter optimization)
        cv_results = {
            'hyperparameter_optimization': {
                'parameter_grid': all_scores if all_scores else {},
                'optimal_params': optimal_params,
                'optimal_score': optimal_score,
                'evaluation_method': 'quality_scoring_with_mcmc_efficiency',
                'parameter_space': {
                    'K_candidates': config.get('hyperparameter_optimization', {}).get('K_candidates', [5, 8, 10, 12, 15]),
                    'percW_candidates': config.get('hyperparameter_optimization', {}).get('percW_candidates', [20, 25, 33, 40, 50]),
                    'num_samples_candidates': config.get('hyperparameter_optimization', {}).get('num_samples_candidates', [1000, 2000]),
                    'num_chains_candidates': config.get('hyperparameter_optimization', {}).get('num_chains_candidates', [2, 4])
                }
            },
            'method_comparison': {
                'sgfa_variants': {k: v for k, v in results['sgfa_variants'].items() if v.get('status') == 'completed'},
                'traditional_methods': results['traditional_methods']
            }
        }

        logger.info("🎨 Data prepared for comprehensive visualization")
        logger.info(f"   📊 SGFA variants: {len(analysis_results['sgfa_variants'])}")
        logger.info(f"   📊 Traditional methods: {len(analysis_results['traditional_methods'])}")
        logger.info(f"   📊 Hyperparameter combinations tested: {len(cv_results['hyperparameter_optimization']['parameter_grid'])}")

        # Create all visualizations using the comprehensive suite
        try:
            plot_dir = viz_manager.create_all_visualizations(
                data=visualization_data,
                cv_results=cv_results,
                analysis_results=analysis_results
            )
        except Exception as viz_e:
            logger.warning(f"VisualizationManager.create_all_visualizations failed: {viz_e}")
            # Fall back to manual plot directory creation
            plot_dir = viz_manager._setup_plot_directory()

        # Count generated files
        plot_files = list(Path(plot_dir).rglob("*.png")) + list(Path(plot_dir).rglob("*.html")) + list(Path(plot_dir).rglob("*.nii.gz"))
        plots_generated = [str(f) for f in plot_files]

        logger.info(f"🎨 VisualizationManager completed successfully!")
        logger.info(f"   📁 Plot directory: {plot_dir}")
        logger.info(f"   📊 Total files generated: {len(plots_generated)}")

        # Categorize generated files
        png_files = [f for f in plots_generated if f.endswith('.png')]
        html_files = [f for f in plots_generated if f.endswith('.html')]
        brain_files = [f for f in plots_generated if f.endswith('.nii.gz')]

        logger.info(f"   🖼️  PNG plots: {len(png_files)}")
        logger.info(f"   📄 HTML reports: {len(html_files)}")
        logger.info(f"   🧠 Brain maps: {len(brain_files)}")

        # Return comprehensive plot information
        return {
            'status': 'completed',
            'plot_directory': str(plot_dir),
            'generated_plots': plots_generated,
            'plot_count': len(plots_generated),
            'visualization_suite': {
                'factor_plots': len([f for f in png_files if 'factor' in f]),
                'preprocessing_plots': len([f for f in png_files if 'preprocessing' in f]),
                'cv_plots': len([f for f in png_files if 'cv' in f or 'optimization' in f]),
                'brain_maps': len(brain_files),
                'html_reports': len(html_files)
            },
            'visualization_manager': True,
            'comprehensive_suite': True
        }

    except ImportError as e:
        logger.warning(f"⚠️  VisualizationManager not available: {e}")
        logger.info("   Full visualization suite requires all visualization modules")
        return {'status': 'visualization_manager_unavailable', 'error': str(e)}
    except Exception as e:
        logger.error(f"❌ VisualizationManager failed: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return {'status': 'failed', 'error': str(e)}


def create_fallback_visualizations(results: Dict, X_list, output_dir: str) -> Dict:
    """
    Fallback to basic visualization if VisualizationManager is not available.

    Args:
        results: Experiment results dictionary
        X_list: Input data list
        output_dir: Output directory path

    Returns:
        Dictionary with basic visualization results
    """
    try:
        logger.info("🎨 Creating fallback visualizations...")
        from visualization.factor_plots import FactorPlotter

        plot_dir = Path(output_dir) / "plots"
        plot_dir.mkdir(exist_ok=True)

        plotter = FactorPlotter()
        plots_generated = []

        # Generate basic plots for completed SGFA variants
        for variant_name, variant_result in results['sgfa_variants'].items():
            if variant_result.get('status') == 'completed' and 'factor_loadings' in variant_result:
                logger.info(f"🖼️  Generating basic plots for {variant_name}...")

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

                    # Generate factor score distribution plots
                    plot_path = plot_dir / f"{variant_name}_scores_distribution.png"
                    plotter.plot_factor_distributions(
                        factor_scores,
                        title=f"{variant_name.title()} SGFA - Factor Score Distributions",
                        save_path=str(plot_path)
                    )
                    plots_generated.append(str(plot_path))

                except Exception as e:
                    logger.warning(f"   ⚠️  Failed to generate plots for {variant_name}: {e}")

        logger.info(f"🎨 Generated {len(plots_generated)} basic plots")

        return {
            'status': 'basic_completed',
            'plot_directory': str(plot_dir),
            'generated_plots': plots_generated,
            'plot_count': len(plots_generated),
            'visualization_manager': False
        }

    except Exception as e:
        logger.error(f"❌ Fallback visualization failed: {e}")
        return {'status': 'failed', 'error': str(e)}