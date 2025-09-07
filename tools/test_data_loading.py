#!/usr/bin/env python
"""Test script to verify data loading works correctly."""

import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, '.')

def test_basic_loading():
    """Test basic data loading."""
    try:
        from data.qmap_pd import load_qmap_pd
        
        # Test with basic configuration
        logger.info("Testing basic data loading...")
        data = load_qmap_pd(
            data_dir="qMAP-PD_data",
            enable_advanced_preprocessing=False
        )
        
        logger.info(f"Data loaded successfully!")
        logger.info(f"Data keys: {list(data.keys())}")
        
        if 'X_list' in data:
            logger.info(f"X_list has {len(data['X_list'])} views")
            for i, X in enumerate(data['X_list']):
                logger.info(f"  View {i}: shape {X.shape}")
        else:
            logger.error("Missing X_list key in data!")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error in basic loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_preprocessing_strategies():
    """Test the specific preprocessing strategies used in experiments."""
    try:
        from data.qmap_pd import load_qmap_pd
        
        # Test the strategies from the experiment
        strategies = {
            'basic': {
                'enable_advanced_preprocessing': False,
                'enable_spatial_processing': False
            },
            'aggressive': {
                'enable_advanced_preprocessing': True,
                'enable_spatial_processing': True,
                'imputation_strategy': 'median',
                'feature_selection_method': 'variance',
                'variance_threshold': 0.1,
                'n_top_features': 1000,
                'spatial_imputation': True,
                'roi_based_selection': True
            }
        }
        
        results = {}
        
        for strategy_name, strategy_config in strategies.items():
            logger.info(f"Testing strategy: {strategy_name}")
            
            try:
                data = load_qmap_pd(
                    data_dir="qMAP-PD_data",
                    **strategy_config
                )
                
                logger.info(f"  Strategy {strategy_name} loaded successfully!")
                logger.info(f"  Data keys: {list(data.keys())}")
                
                if 'X_list' in data:
                    logger.info(f"  X_list has {len(data['X_list'])} views")
                    for i, X in enumerate(data['X_list']):
                        logger.info(f"    View {i}: shape {X.shape}")
                    results[strategy_name] = data
                else:
                    logger.error(f"  Missing X_list key in {strategy_name}!")
                    
            except Exception as e:
                logger.error(f"  Error in {strategy_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return results
        
    except Exception as e:
        logger.error(f"Error in strategy testing: {e}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    logger.info("=== Testing Data Loading ===")
    
    # Test basic loading
    basic_success = test_basic_loading()
    
    if basic_success:
        logger.info("\n=== Testing Preprocessing Strategies ===")
        strategy_results = test_advanced_preprocessing_strategies()
        
        if strategy_results:
            logger.info(f"\nSuccess: Successfully tested {len(strategy_results)} strategies!")
            for name in strategy_results.keys():
                logger.info(f"  PASSED: {name}")
        else:
            logger.error("FAILED: No strategies worked")
    else:
        logger.error("FAILED: Basic loading failed")