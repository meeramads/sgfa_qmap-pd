#!/usr/bin/env python3
"""
Test script for VisualizationManager integration in remote workstation pipeline.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_visualization_integration():
    """Test the VisualizationManager integration functionality."""
    print("Testing VisualizationManager integration...")

    # Mock data similar to what the pipeline would have
    np.random.seed(42)
    n_subjects = 30

    # Create multi-view data
    X_list = [
        np.random.randn(n_subjects, 500),   # View 1: structural
        np.random.randn(n_subjects, 1000),  # View 2: functional
        np.random.randn(n_subjects, 200),   # View 3: diffusion
        np.random.randn(n_subjects, 50)     # View 4: clinical
    ]

    # Mock results structure
    results = {
        'sgfa_variants': {
            'standard': {
                'status': 'completed',
                'factor_loadings': {
                    'mean': np.random.randn(sum(X.shape[1] for X in X_list), 10).tolist()
                },
                'factor_scores': {
                    'mean': np.random.randn(n_subjects, 10).tolist()
                },
                'n_factors': 10,
                'duration_minutes': 5.2,
                'config': {'use_sparse': True, 'use_group': True}
            },
            'sparse_only': {
                'status': 'completed',
                'factor_loadings': {
                    'mean': np.random.randn(sum(X.shape[1] for X in X_list), 8).tolist()
                },
                'factor_scores': {
                    'mean': np.random.randn(n_subjects, 8).tolist()
                },
                'n_factors': 8,
                'duration_minutes': 3.8,
                'config': {'use_sparse': True, 'use_group': False}
            }
        },
        'traditional_methods': {
            'pca': {
                'status': 'completed',
                'n_components': 10,
                'explained_variance_ratio': 0.85,
                'duration_seconds': 0.5
            },
            'ica': {
                'status': 'completed',
                'n_components': 10,
                'duration_seconds': 2.1
            }
        }
    }

    # Mock configuration
    config = {
        'data': {'data_dir': './qMAP-PD_data'},
        'hyperparameter_optimization': {
            'K_candidates': [5, 8, 10, 12],
            'percW_candidates': [20, 25, 33, 40],
            'num_samples_candidates': [1000, 2000],
            'num_chains_candidates': [2, 4]
        }
    }

    # Mock optimal parameters
    optimal_params = {'K': 10, 'percW': 33, 'num_samples': 1000, 'num_chains': 2}
    optimal_score = 0.642
    all_scores = {'K10_percW33_samples1000_chains2': {'score': 0.642}}

    output_dir = "./test_visualization_output"
    Path(output_dir).mkdir(exist_ok=True)

    print(f"Mock data created: {len(X_list)} views")
    print(f"Mock results: {len(results['sgfa_variants'])} SGFA variants, {len(results['traditional_methods'])} traditional methods")

    # Test the visualization integration
    try:
        from remote_workstation.visualization_integration import create_comprehensive_visualizations, create_fallback_visualizations

        print("\n=== Testing Comprehensive VisualizationManager ===")

        # Try comprehensive visualization
        viz_results = create_comprehensive_visualizations(
            results=results,
            X_list=X_list,
            optimal_params=optimal_params,
            optimal_score=optimal_score,
            config=config,
            output_dir=output_dir,
            all_scores=all_scores
        )

        print(f"Visualization result status: {viz_results['status']}")

        if viz_results['status'] == 'completed':
            print(f"✅ Comprehensive visualization succeeded!")
            print(f"   📁 Plot directory: {viz_results['plot_directory']}")
            print(f"   📊 Total files: {viz_results['plot_count']}")

            if 'visualization_suite' in viz_results:
                suite = viz_results['visualization_suite']
                print(f"   🎨 Visualization breakdown:")
                print(f"      📊 Factor plots: {suite.get('factor_plots', 0)}")
                print(f"      📈 Preprocessing plots: {suite.get('preprocessing_plots', 0)}")
                print(f"      📉 CV plots: {suite.get('cv_plots', 0)}")
                print(f"      🧠 Brain maps: {suite.get('brain_maps', 0)}")
                print(f"      📄 HTML reports: {suite.get('html_reports', 0)}")

        elif viz_results['status'] == 'visualization_manager_unavailable':
            print("⚠️  VisualizationManager not available, testing fallback...")

            fallback_results = create_fallback_visualizations(results, X_list, output_dir)
            print(f"Fallback result status: {fallback_results['status']}")

            if fallback_results['status'] == 'basic_completed':
                print(f"✅ Fallback visualization succeeded!")
                print(f"   📁 Plot directory: {fallback_results['plot_directory']}")
                print(f"   📊 Total files: {fallback_results['plot_count']}")
            else:
                print(f"❌ Fallback visualization failed: {fallback_results.get('error', 'Unknown error')}")

        else:
            print(f"❌ Visualization failed: {viz_results.get('error', 'Unknown error')}")

        # Simulate what the pipeline would log
        print(f"\n=== Pipeline Visualization Summary (as it would appear) ===")

        if viz_results['status'] == 'completed':
            plot_count = viz_results.get('plot_count', 0)
            print(f"🎨 VISUALIZATION SUMMARY:")
            print(f"   ✅ Generated {plot_count} plots")
            print(f"   📁 Plot directory: {viz_results['plot_directory']}")

            if viz_results.get('visualization_manager', False):
                print(f"   🎨 Comprehensive VisualizationManager suite used")
                viz_suite = viz_results.get('visualization_suite', {})
                if viz_suite:
                    print(f"   📊 Factor plots: {viz_suite.get('factor_plots', 0)}")
                    print(f"   📈 Preprocessing plots: {viz_suite.get('preprocessing_plots', 0)}")
                    print(f"   📉 CV/optimization plots: {viz_suite.get('cv_plots', 0)}")
                    print(f"   🧠 Brain maps: {viz_suite.get('brain_maps', 0)}")
                    print(f"   📄 HTML reports: {viz_suite.get('html_reports', 0)}")
                if viz_results.get('comprehensive_suite', False):
                    print(f"   ✅ Full visualization capabilities utilized")
            else:
                print(f"   📊 Basic visualization used (VisualizationManager unavailable)")

        return viz_results

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_visualization_integration()

    if result and result['status'] in ['completed', 'basic_completed']:
        print(f"\n🎯 Visualization integration test successful!")
        print(f"   The remote workstation pipeline now supports comprehensive visualization")
        print(f"   Status: {result['status']}")
        if result.get('visualization_manager', False):
            print(f"   ✅ Full VisualizationManager suite available")
        else:
            print(f"   📊 Fallback visualization available")
    else:
        print(f"\n⚠️  Test completed with issues")
        if result:
            print(f"   Status: {result['status']}")
            print(f"   Error: {result.get('error', 'See output above')}")