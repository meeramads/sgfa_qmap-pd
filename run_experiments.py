#!/usr/bin/env python
"""
Remote Workstation Experiment Runner
Runs the complete experimental framework optimized for university GPU resources.

This modular version imports experiments from separate modules for better organization.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

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

# Import experiment functions
from experiments.data_validation import run_data_validation
from experiments.method_comparison import run_method_comparison
from experiments.performance_benchmarks import run_performance_benchmarks
from experiments.sensitivity_analysis import run_sensitivity_analysis

def load_config(config_path="config.yaml"):
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

def main():
    """Main experimental pipeline."""
    parser = argparse.ArgumentParser(description="Run Remote Workstation experimental framework")
    parser.add_argument("--config", default="config.yaml",
                       help="Configuration file path")
    parser.add_argument("--experiments", nargs="+",
                       choices=["data_validation", "method_comparison",
                               "performance_benchmarks", "sensitivity_analysis", "all"],
                       default=["all"], help="Experiments to run")
    parser.add_argument("--data-dir", help="Override data directory")
    parser.add_argument("--unified-results", action="store_true", default=True,
                       help="Save all results in a single timestamped folder (default: True)")
    parser.add_argument("--shared-data", action="store_true", default=True,
                       help="Use shared data pipeline for efficiency (default: True)")
    parser.add_argument("--independent-mode", action="store_true", default=False,
                       help="Force independent data loading for troubleshooting (overrides --shared-data)")

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

    # Determine execution mode
    use_shared_data = args.shared_data and not args.independent_mode
    if args.independent_mode:
        logger.info("🔧 Using INDEPENDENT MODE - each experiment loads its own data (for troubleshooting)")
    elif use_shared_data:
        logger.info("🔗 Using SHARED DATA MODE - efficient pipeline with data reuse")
    else:
        logger.info("🔧 Using INDEPENDENT MODE - shared data disabled")

    # Initialize pipeline context for data sharing
    pipeline_context = {
        'X_list': None,
        'preprocessing_info': None,
        'data_strategy': None,
        'shared_mode': use_shared_data,
        'memory_usage_mb': 0
    }

    logger.info(f"🔄 Pipeline context initialized (shared_mode: {use_shared_data})")

    # Run experiments sequentially, passing context through config
    if "data_validation" in experiments_to_run:
        logger.info("🔍 Starting Data Validation Experiment...")
        results['data_validation'] = run_data_validation(config)

        # Update pipeline context with results from data validation
        if results['data_validation'] and hasattr(results['data_validation'], 'model_results'):
            model_results = results['data_validation'].model_results
            if 'preprocessed_data' in model_results:
                pipeline_context['X_list'] = model_results['preprocessed_data'].get('X_list')
                pipeline_context['preprocessing_info'] = model_results['preprocessed_data'].get('preprocessing_info')
                pipeline_context['data_strategy'] = model_results.get('data_strategy', 'unknown')

        # Log context update
        if pipeline_context['X_list'] is not None:
            logger.info(f"📊 Data validation loaded data: {len(pipeline_context['X_list'])} views")
            logger.info(f"   Strategy: {pipeline_context.get('data_strategy', 'unknown')}")

    if "method_comparison" in experiments_to_run:
        logger.info("🧠 Starting Method Comparison Experiment...")
        exp_config = config.copy()
        if pipeline_context['X_list'] is not None and use_shared_data:
            logger.info("   → Using shared data from data_validation")
            exp_config['_shared_data'] = {
                'X_list': pipeline_context['X_list'],
                'preprocessing_info': pipeline_context['preprocessing_info'],
                'mode': 'shared'
            }
        results['method_comparison'] = run_method_comparison(exp_config)

    if "performance_benchmarks" in experiments_to_run:
        logger.info("⚡ Starting Performance Benchmark Experiment...")
        exp_config = config.copy()
        if pipeline_context['X_list'] is not None and use_shared_data:
            logger.info("   → Using shared data from previous experiments")
            exp_config['_shared_data'] = {
                'X_list': pipeline_context['X_list'],
                'preprocessing_info': pipeline_context['preprocessing_info'],
                'mode': 'shared'
            }
        results['performance_benchmarks'] = run_performance_benchmarks(exp_config)

    if "sensitivity_analysis" in experiments_to_run:
        logger.info("📊 Starting Sensitivity Analysis Experiment...")
        exp_config = config.copy()
        if pipeline_context['X_list'] is not None and use_shared_data:
            logger.info("   → Using shared data from previous experiments")
            exp_config['_shared_data'] = {
                'X_list': pipeline_context['X_list'],
                'preprocessing_info': pipeline_context['preprocessing_info'],
                'mode': 'shared'
            }
        results['sensitivity_analysis'] = run_sensitivity_analysis(exp_config)

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