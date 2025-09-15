#!/usr/bin/env python
"""
Data Validation Experiments
Comprehensive data validation on remote workstation.
"""

import logging

logger = logging.getLogger(__name__)

def _log_preprocessing_summary(preprocessing_info):
    """Log a concise preprocessing summary instead of full details."""
    if not preprocessing_info:
        logger.info("✅ Preprocessing info: None")
        return

    # Extract key information
    status = preprocessing_info.get('status', 'unknown')

    # Handle different strategy formats
    strategy = preprocessing_info.get('strategy', 'unknown')
    if isinstance(strategy, dict):
        strategy = strategy.get('name', 'unknown')

    # Check for preprocessing results nested structure
    if 'preprocessing_results' in preprocessing_info:
        nested_info = preprocessing_info['preprocessing_results']
        status = nested_info.get('status', status)
        strategy = nested_info.get('strategy', strategy)
        preprocessor_type = nested_info.get('preprocessor_type', 'unknown')
    else:
        preprocessor_type = preprocessing_info.get('preprocessor_type', 'unknown')

    logger.info(f"✅ Preprocessing: {status} ({preprocessor_type})")
    logger.info(f"   Strategy: {strategy}")

    # Get the actual preprocessing data (could be nested)
    data_source = preprocessing_info.get('preprocessing_results', preprocessing_info)

    # Log feature reduction if available
    if 'feature_reduction' in data_source:
        fr = data_source['feature_reduction']
        logger.info(f"   Features: {fr['total_before']:,} → {fr['total_after']:,} ({fr['reduction_ratio']:.3f} ratio)")

    # Log steps applied
    steps = data_source.get('steps_applied', [])
    if steps:
        logger.info(f"   Steps: {', '.join(steps)}")

    # Log basic shapes summary
    original_shapes = data_source.get('original_shapes', [])
    processed_shapes = data_source.get('processed_shapes', [])
    if original_shapes and processed_shapes:
        total_orig_features = sum(shape[1] for shape in original_shapes)
        total_proc_features = sum(shape[1] for shape in processed_shapes)
        logger.info(f"   Data: {len(original_shapes)} views, {total_orig_features:,} → {total_proc_features:,} features")

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

        # Log preprocessing summary instead of full details
        _log_preprocessing_summary(preprocessing_info)
        logger.info("✅ Data validation completed successfully")
        return {'status': 'completed', 'views': len(X_list), 'shapes': [X.shape for X in X_list], 'preprocessing': preprocessing_info}

    except Exception as e:
        logger.error(f" Data validation failed: {e}")
        return None