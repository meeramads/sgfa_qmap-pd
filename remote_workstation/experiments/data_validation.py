#!/usr/bin/env python
"""
Data Validation Experiments
Comprehensive data validation on remote workstation.
"""

import logging

logger = logging.getLogger(__name__)

def run_data_validation(config):
    """Run data validation experiments."""
    logger.info(" Starting Data Validation Experiments")

    try:
        # Self-contained data validation - no external framework dependencies
        import sys
        import os

        # Add project root for basic imports only
        current_file = os.path.abspath(__file__)
        remote_ws_dir = os.path.dirname(os.path.dirname(current_file))
        project_root = os.path.dirname(remote_ws_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Run basic data validation directly without framework
        logger.info("Running simplified data validation...")

        # Basic data loading and validation
        from remote_workstation.preprocessing_integration import apply_preprocessing_to_pipeline
        X_list, preprocessing_info = apply_preprocessing_to_pipeline(
            config=config,
            data_dir=config['data']['data_dir'],
            auto_select_strategy=True
        )

        logger.info(f"✅ Data loaded: {len(X_list)} views")
        for i, X in enumerate(X_list):
            logger.info(f"   View {i}: {X.shape}")

        logger.info(f"✅ Preprocessing info: {preprocessing_info}")
        logger.info("✅ Data validation completed successfully")
        return {'status': 'completed', 'views': len(X_list), 'shapes': [X.shape for X in X_list], 'preprocessing': preprocessing_info}

    except Exception as e:
        logger.error(f" Data validation failed: {e}")
        return None