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