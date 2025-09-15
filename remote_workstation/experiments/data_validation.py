#!/usr/bin/env python
"""
Data Validation Experiments
Comprehensive data validation on remote workstation.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run_data_validation(config):
    """Run data validation experiments."""
    logger.info(" Starting Data Validation Experiments")

    try:
        # Add project root to path for framework imports
        import sys
        import os

        # Get the project root (parent of remote_workstation)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Import framework using direct module loading to avoid relative import issues
        import importlib.util

        # Import framework directly
        framework_path = os.path.join(project_root, 'experiments', 'framework.py')
        spec = importlib.util.spec_from_file_location("framework", framework_path)
        framework_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(framework_module)
        ExperimentFramework = framework_module.ExperimentFramework
        ExperimentConfig = framework_module.ExperimentConfig

        # Import data validation experiments
        data_val_path = os.path.join(project_root, 'experiments', 'data_validation.py')
        spec = importlib.util.spec_from_file_location("data_validation", data_val_path)
        data_val_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_val_module)
        DataValidationExperiments = data_val_module.DataValidationExperiments

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