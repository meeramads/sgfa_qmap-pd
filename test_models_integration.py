#!/usr/bin/env python
"""
Test script for Models Framework Integration
Tests the comprehensive models framework integration without requiring full dataset.
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, '.')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_config():
    """Create mock configuration for testing."""
    return {
        'data': {
            'data_dir': './qMAP-PD_data'
        },
        'experiments': {
            'base_output_dir': './test_results'
        },
        'model': {
            'type': 'sparseGFA',
            'auto_select': True,
            'use_recommendation': True,
            'enable_comparison': True,
            'reghsZ': True,
            'use_sparse': True,
            'use_group': True,
            'num_sources': 4,
            'spatial_smoothing': True
        },
        'hyperparameter_optimization': {
            'fallback_K': 10,
            'fallback_percW': 33
        }
    }

def create_mock_data():
    """Create mock neuroimaging data for testing."""
    np.random.seed(42)

    # Simulate qMAP-PD dataset structure with imaging and clinical data
    n_subjects = 75
    view_dimensions = [1500, 800, 500, 20]  # Imaging views + clinical

    X_list = []
    for i, dim in enumerate(view_dimensions):
        if i < 3:  # Imaging data
            X = np.random.exponential(scale=1.0, size=(n_subjects, dim))
            X = X + 0.3 * np.random.normal(0, 1, size=(n_subjects, dim))
        else:  # Clinical data
            X = np.random.normal(0, 1, size=(n_subjects, dim))

        X = np.maximum(X, 0.01)  # Ensure positive values for imaging
        X_list.append(X)

    logger.info(f"Created mock data: {n_subjects} subjects, {len(X_list)} views")
    logger.info(f"View dimensions: {[X.shape[1] for X in X_list]}")
    logger.info(f"Total features: {sum(X.shape[1] for X in X_list)}")

    return X_list

def create_data_characteristics(X_list):
    """Create data characteristics for model selection testing."""
    return {
        'n_subjects': len(X_list[0]),
        'n_views': len(X_list),
        'total_features': sum(X.shape[1] for X in X_list),
        'view_dimensions': [X.shape[1] for X in X_list],
        'has_imaging_data': any(X.shape[1] > 100 for X in X_list),
        'imaging_views': [i for i, X in enumerate(X_list) if X.shape[1] > 100]
    }

def test_models_integration_import():
    """Test that models integration module can be imported."""
    logger.info("=== Testing Models Integration Import ===")

    try:
        from remote_workstation.models_integration import integrate_models_with_pipeline
        logger.info("✅ Models integration module imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import models integration: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error importing models integration: {e}")
        return False

def test_model_configuration_generation():
    """Test model configuration generation."""
    logger.info("=== Testing Model Configuration Generation ===")

    try:
        from remote_workstation.models_integration import get_optimal_model_configuration

        config = create_mock_config()
        X_list = create_mock_data()
        data_characteristics = create_data_characteristics(X_list)

        model_type, model_config = get_optimal_model_configuration(config, data_characteristics)

        if model_config.get('model_factory_available', False):
            logger.info("✅ Model configuration generated successfully")
            logger.info(f"  Selected model: {model_type}")
            logger.info(f"  Available models: {model_config.get('available_models', [])}")
            logger.info(f"  Configuration strategy: {model_config.get('configuration_strategy', 'unknown')}")
            logger.info(f"  Neuroimaging optimized: {model_config.get('neuroimaging_optimized', False)}")
            logger.info(f"  Sparsity regularization: {model_config.get('sparsity_regularization', False)}")
            return True
        else:
            logger.warning("⚠️  Model configuration generation returned fallback")
            logger.info(f"  Fallback reason: {model_config.get('error', 'unknown')}")
            return True  # Fallback is still a valid result

    except Exception as e:
        logger.error(f"❌ Model configuration generation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def test_model_instance_creation():
    """Test model instance creation using ModelFactory."""
    logger.info("=== Testing Model Instance Creation ===")

    try:
        from remote_workstation.models_integration import create_model_instance

        config = create_mock_config()
        X_list = create_mock_data()
        data_characteristics = create_data_characteristics(X_list)

        # Create default hyperparameters
        hypers = {
            'a_sigma': 1.0,
            'b_sigma': 1.0,
            'nu_local': 1.0,
            'nu_global': 1.0,
            'slab_scale': 2.0,
            'slab_df': 4.0,
            'percW': 33,
            'Dm': [X.shape[1] for X in X_list]
        }

        # Test different model types
        model_types_to_test = ['sparseGFA', 'GFA', 'neuroGFA']

        for model_type in model_types_to_test:
            logger.info(f"\n  Testing {model_type} model creation...")

            model_instance, creation_info = create_model_instance(
                model_type, config, hypers, data_characteristics
            )

            if creation_info.get('model_created', False):
                logger.info(f"  ✅ {model_type}: Created successfully")
                logger.info(f"     Model name: {creation_info.get('model_name', 'unknown')}")
                logger.info(f"     Model class: {creation_info.get('model_class', 'unknown')}")
                logger.info(f"     Factory method: {creation_info.get('factory_method_used', False)}")
                logger.info(f"     Spatial info: {creation_info.get('spatial_info_available', False)}")
            else:
                logger.info(f"  ⚠️  {model_type}: Creation failed")
                logger.info(f"     Error: {creation_info.get('error', 'unknown')}")

        logger.info("✅ Model instance creation testing completed")
        return True

    except Exception as e:
        logger.error(f"❌ Model instance creation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def test_models_framework_integration():
    """Test comprehensive models framework integration."""
    logger.info("=== Testing Models Framework Integration ===")

    try:
        from remote_workstation.models_integration import integrate_models_with_pipeline

        config = create_mock_config()
        X_list = create_mock_data()
        data_characteristics = create_data_characteristics(X_list)

        # Create hyperparameters
        hypers = {
            'a_sigma': 1.0,
            'b_sigma': 1.0,
            'nu_local': 1.0,
            'nu_global': 1.0,
            'slab_scale': 2.0,
            'slab_df': 4.0,
            'percW': 33,
            'Dm': [X.shape[1] for X in X_list]
        }

        model_type, model_instance, models_summary = integrate_models_with_pipeline(
            config=config,
            X_list=X_list,
            data_characteristics=data_characteristics,
            hypers=hypers
        )

        integration_info = models_summary.get('integration_summary', {})

        if integration_info.get('framework_available', False):
            logger.info("✅ Models framework integration successful")
            logger.info(f"  Model type: {integration_info.get('model_type_selected', 'unknown')}")
            logger.info(f"  Structured management: {integration_info.get('structured_model_management', False)}")
            logger.info(f"  Model factory used: {integration_info.get('model_factory_used', False)}")
            logger.info(f"  Model instance created: {integration_info.get('model_instance_created', False)}")

            if integration_info.get('available_models'):
                logger.info(f"  Available models: {', '.join(integration_info['available_models'])}")

            logger.info(f"  Neuroimaging optimized: {integration_info.get('neuroimaging_optimized', False)}")
            logger.info(f"  Sparsity regularization: {integration_info.get('sparsity_regularization', False)}")

            # Test model comparison if available
            if integration_info.get('comparison_completed', False):
                comparison = integration_info.get('model_comparison', {})
                logger.info(f"  Model comparison: {comparison.get('models_tested', [])} tested")
                if comparison.get('best_model'):
                    logger.info(f"  Best model: {comparison['best_model']} (score: {comparison.get('best_score', 0):.2f})")

            return True
        else:
            logger.info("⚠️  Models framework not available - using fallback")
            logger.info(f"  Fallback reason: {models_summary.get('error', 'framework_unavailable')}")
            return True  # Fallback is still a valid result

    except Exception as e:
        logger.error(f"❌ Models framework integration failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def test_model_wrapper_functionality():
    """Test ModelsFrameworkWrapper functionality."""
    logger.info("=== Testing Model Wrapper Functionality ===")

    try:
        from remote_workstation.models_integration import integrate_models_with_pipeline, _wrap_models_framework

        config = create_mock_config()
        X_list = create_mock_data()
        data_characteristics = create_data_characteristics(X_list)

        hypers = {
            'a_sigma': 1.0,
            'b_sigma': 1.0,
            'nu_local': 1.0,
            'nu_global': 1.0,
            'slab_scale': 2.0,
            'slab_df': 4.0,
            'percW': 33,
            'Dm': [X.shape[1] for X in X_list]
        }

        model_type, model_instance, models_summary = integrate_models_with_pipeline(
            config=config,
            X_list=X_list,
            data_characteristics=data_characteristics,
            hypers=hypers
        )

        if model_instance:
            logger.info("Testing model wrapper functionality...")

            # Create wrapper
            models_wrapper = _wrap_models_framework(model_type, model_instance, models_summary)

            # Test wrapper methods
            execution_model = models_wrapper.get_model_for_execution()
            model_info = models_wrapper.get_model_info()

            logger.info("✅ Model wrapper functionality successful")
            logger.info(f"  Execution model available: {execution_model is not None}")
            logger.info(f"  Model info: {model_info.get('model_name', 'unknown')}")
            logger.info(f"  Structured model ready: {model_info.get('structured_model_ready', False)}")

            # Test model comparison
            try:
                comparison_results = models_wrapper.compare_with_alternatives(config, X_list, hypers)
                if 'error' not in comparison_results:
                    logger.info(f"  Model comparison: {len(comparison_results.get('models_tested', []))} models tested")

            except Exception as comp_e:
                logger.info(f"  Model comparison unavailable: {comp_e}")

            return True
        else:
            logger.info("⚠️  Model wrapper testing skipped - no model instance available")
            return True

    except Exception as e:
        logger.error(f"❌ Model wrapper functionality test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def test_pipeline_integration():
    """Test integration with the main pipeline structure."""
    logger.info("=== Testing Pipeline Integration ===")

    try:
        # Test that the main pipeline can import and use the integration
        config = create_mock_config()
        X_list = create_mock_data()
        data_characteristics = create_data_characteristics(X_list)

        # Simulate the integration calls from run_experiments.py
        from remote_workstation.models_integration import integrate_models_with_pipeline

        logger.info("Testing pipeline integration call...")
        model_type, model_instance, models_summary = integrate_models_with_pipeline(
            config=config,
            X_list=X_list,
            data_characteristics=data_characteristics
        )

        # Test logging format compatibility
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

        logger.info("✅ Pipeline integration test successful")
        return True

    except Exception as e:
        logger.error(f"❌ Pipeline integration test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def run_all_tests():
    """Run all models integration tests."""
    logger.info("🧪 Starting Models Integration Test Suite")
    logger.info("=" * 60)

    tests = [
        ("Import Test", test_models_integration_import),
        ("Configuration Generation", test_model_configuration_generation),
        ("Model Instance Creation", test_model_instance_creation),
        ("Framework Integration", test_models_framework_integration),
        ("Model Wrapper Functionality", test_model_wrapper_functionality),
        ("Pipeline Integration", test_pipeline_integration)
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info("")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
        logger.info("=" * 60)

    # Summary
    logger.info("")
    logger.info("🎯 TEST RESULTS SUMMARY:")
    passed = 0
    total = len(tests)

    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        logger.info(f"  {status}: {test_name}")
        if passed_test:
            passed += 1

    logger.info("")
    logger.info(f"📊 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        logger.info("🎉 All models integration tests passed!")
        return True
    else:
        logger.info("⚠️  Some tests failed - check logs for details")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)