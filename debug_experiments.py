#!/usr/bin/env python
"""
Quick Debug Experiment Runner
Individual experiment runners for debugging and development.

Usage:
    python debug_experiments.py data_validation
    python debug_experiments.py sgfa_parameter_comparison
    python debug_experiments.py model_comparison
    python debug_experiments.py performance_benchmarks
    python debug_experiments.py sensitivity_analysis
    python debug_experiments.py clinical_validation
    python debug_experiments.py all
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, ".")

# Set up simple logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_debug_config():
    """Load debug configuration."""
    config_path = "config_debug.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded debug configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load debug config: {e}")
        sys.exit(1)


def run_data_validation_debug():
    """Run minimal data validation experiment."""
    logger.info("🔍 Running DEBUG: Data Validation")

    from experiments.data_validation import run_data_validation

    config = load_debug_config()
    start_time = time.time()

    try:
        result = run_data_validation(config)
        duration = time.time() - start_time

        if result:
            logger.info(f"✅ Data validation completed in {duration:.2f}s")
            logger.info(f"   Status: {getattr(result, 'success', 'Unknown')}")
        else:
            logger.error("❌ Data validation failed")

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Data validation failed after {duration:.2f}s: {e}")
        raise


def run_sgfa_parameter_comparison_debug():
    """Run minimal SGFA parameter comparison."""
    logger.info("🔬 Running DEBUG: SGFA Parameter Comparison")

    from experiments.sgfa_parameter_comparison import run_method_comparison

    config = load_debug_config()
    start_time = time.time()

    try:
        result = run_method_comparison(config)
        duration = time.time() - start_time

        if result:
            logger.info(f"✅ SGFA parameter comparison completed in {duration:.2f}s")
            logger.info(f"   Status: {getattr(result, 'success', 'Unknown')}")
        else:
            logger.error("❌ SGFA parameter comparison failed")

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ SGFA parameter comparison failed after {duration:.2f}s: {e}")
        raise


def run_model_comparison_debug():
    """Run minimal model comparison."""
    logger.info("🧠 Running DEBUG: Model Comparison")

    from experiments.model_comparison import run_model_comparison

    config = load_debug_config()
    start_time = time.time()

    try:
        result = run_model_comparison(config)
        duration = time.time() - start_time

        if result:
            logger.info(f"✅ Model comparison completed in {duration:.2f}s")
            logger.info(f"   Status: {getattr(result, 'success', 'Unknown')}")
        else:
            logger.error("❌ Model comparison failed")

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Model comparison failed after {duration:.2f}s: {e}")
        raise


def run_performance_benchmarks_debug():
    """Run minimal performance benchmarks."""
    logger.info("⚡ Running DEBUG: Performance Benchmarks")

    from experiments.performance_benchmarks import run_performance_benchmarks

    config = load_debug_config()
    start_time = time.time()

    try:
        result = run_performance_benchmarks(config)
        duration = time.time() - start_time

        if result:
            logger.info(f"✅ Performance benchmarks completed in {duration:.2f}s")
            logger.info(f"   Status: {getattr(result, 'success', 'Unknown')}")
        else:
            logger.error("❌ Performance benchmarks failed")

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Performance benchmarks failed after {duration:.2f}s: {e}")
        raise


def run_sensitivity_analysis_debug():
    """Run minimal sensitivity analysis."""
    logger.info("📊 Running DEBUG: Sensitivity Analysis")

    from experiments.sensitivity_analysis import run_sensitivity_analysis

    config = load_debug_config()
    start_time = time.time()

    try:
        result = run_sensitivity_analysis(config)
        duration = time.time() - start_time

        if result:
            logger.info(f"✅ Sensitivity analysis completed in {duration:.2f}s")
            logger.info(f"   Status: {getattr(result, 'success', 'Unknown')}")
        else:
            logger.error("❌ Sensitivity analysis failed")

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Sensitivity analysis failed after {duration:.2f}s: {e}")
        raise


def run_clinical_validation_debug():
    """Run minimal clinical validation."""
    logger.info("🏥 Running DEBUG: Clinical Validation")

    from experiments.clinical_validation import run_clinical_validation

    config = load_debug_config()
    start_time = time.time()

    try:
        result = run_clinical_validation(config)
        duration = time.time() - start_time

        if result:
            logger.info(f"✅ Clinical validation completed in {duration:.2f}s")
            logger.info(f"   Status: {getattr(result, 'success', 'Unknown')}")
        else:
            logger.error("❌ Clinical validation failed")

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Clinical validation failed after {duration:.2f}s: {e}")
        raise


def run_all_debug():
    """Run all debug experiments sequentially."""
    logger.info("🚀 Running ALL DEBUG EXPERIMENTS")

    experiments = [
        ("data_validation", run_data_validation_debug),
        ("sgfa_parameter_comparison", run_sgfa_parameter_comparison_debug),
        ("model_comparison", run_model_comparison_debug),
        ("performance_benchmarks", run_performance_benchmarks_debug),
        ("sensitivity_analysis", run_sensitivity_analysis_debug),
        ("clinical_validation", run_clinical_validation_debug),
    ]

    total_start = time.time()
    results = {}

    for exp_name, exp_func in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting {exp_name}")
        logger.info(f"{'='*60}")

        try:
            exp_func()
            results[exp_name] = "success"
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            results[exp_name] = "failed"
            # Continue with other experiments

    total_duration = time.time() - total_start

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DEBUG EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total duration: {total_duration:.2f}s")

    successful = len([r for r in results.values() if r == "success"])
    failed = len([r for r in results.values() if r == "failed"])

    logger.info(f"Successful: {successful}/{len(experiments)}")
    logger.info(f"Failed: {failed}/{len(experiments)}")

    for exp_name, status in results.items():
        status_emoji = "✅" if status == "success" else "❌"
        logger.info(f"  {status_emoji} {exp_name}: {status}")


def main():
    """Main debug runner."""
    parser = argparse.ArgumentParser(
        description="Quick Debug Experiment Runner"
    )
    parser.add_argument(
        "experiment",
        choices=[
            "data_validation",
            "sgfa_parameter_comparison",
            "model_comparison",
            "performance_benchmarks",
            "sensitivity_analysis",
            "clinical_validation",
            "all"
        ],
        help="Experiment to run"
    )

    args = parser.parse_args()

    # Create debug results directory
    debug_dir = Path("debug_results")
    debug_dir.mkdir(exist_ok=True)

    logger.info(f"🐛 Starting debug mode for: {args.experiment}")
    logger.info(f"📁 Debug results will be saved to: {debug_dir}")

    # Map experiment names to functions
    experiment_map = {
        "data_validation": run_data_validation_debug,
        "sgfa_parameter_comparison": run_sgfa_parameter_comparison_debug,
        "model_comparison": run_model_comparison_debug,
        "performance_benchmarks": run_performance_benchmarks_debug,
        "sensitivity_analysis": run_sensitivity_analysis_debug,
        "clinical_validation": run_clinical_validation_debug,
        "all": run_all_debug,
    }

    # Run selected experiment
    try:
        experiment_map[args.experiment]()
        logger.info(f"🎉 Debug session completed successfully!")
    except Exception as e:
        logger.error(f"💥 Debug session failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()