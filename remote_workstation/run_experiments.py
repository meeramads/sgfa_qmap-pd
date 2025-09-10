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
        
        if len([d for d in devices if d.device_kind == 'gpu']) == 0:
            logger.warning("  No GPU devices found - will use CPU (slower)")
        else:
            logger.info(" GPU devices available for acceleration")
            
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
        quality_result = validator.assess_data_quality(exp_config)
        
        # Preprocessing comparison
        logger.info("   Running preprocessing comparison...")
        preprocessing_result = validator.compare_preprocessing_strategies(exp_config)
        
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
        
        comparator = MethodComparisonExperiments(framework)
        result = comparator.compare_models(exp_config)
        
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
        
        benchmarker = PerformanceBenchmarkExperiments(framework)
        result = benchmarker.run_benchmarks(exp_config)
        
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
        
        analyzer = SensitivityAnalysisExperiments(framework)
        result = analyzer.analyze_sensitivity(exp_config)
        
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
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override data directory if provided
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
        logger.info(f"Using data directory: {args.data_dir}")
    
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
    
    # Save summary
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