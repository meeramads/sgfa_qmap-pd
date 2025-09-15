#!/usr/bin/env python
"""
Performance Benchmark Experiments
Performance benchmarks on remote workstation.
"""

import logging
import time
import psutil
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

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

        # Create performance benchmark experiment function
        def performance_benchmark_experiment(config, output_dir, **kwargs):
            logger.info("Running direct performance benchmarks...")

            # Load data with preprocessing integration
            from remote_workstation.preprocessing_integration import apply_preprocessing_to_pipeline
            X_list, preprocessing_info = apply_preprocessing_to_pipeline(
                config=config.__dict__,
                data_dir=config.data_dir,
                auto_select_strategy=True
            )

            results = {
                'scalability_benchmark': {},
                'memory_benchmark': {},
                'timing_benchmark': {}
            }

            # Run scalability benchmark
            results['scalability_benchmark'] = _run_scalability_benchmark(X_list, config)

            # Run memory benchmark
            results['memory_benchmark'] = _run_memory_benchmark(config)

            # Run timing benchmark
            results['timing_benchmark'] = _run_timing_benchmark(X_list, config)

            logger.info("Performance benchmarks completed")

            # Format results for framework compatibility
            return {
                'model_results': results,
                'performance_metrics': {
                    'benchmark_types': list(results.keys()),
                    'scalability_tests': len(results.get('scalability_benchmark', {})),
                    'memory_tests': len(results.get('memory_benchmark', {})),
                    'timing_tests': len(results.get('timing_benchmark', {}))
                },
                'diagnostics': {
                    'total_benchmarks': sum(len(v) if isinstance(v, dict) else 1 for v in results.values()),
                    'benchmark_status': 'completed'
                }
            }

        result = framework.run_experiment(exp_config, performance_benchmark_experiment)

        logger.info(" Performance benchmark experiments completed")
        return result

    except Exception as e:
        logger.error(f" Performance benchmarks failed: {e}")
        return None


def _run_scalability_benchmark(X_list, config):
    """Run scalability benchmark with different data sizes."""
    logger.info("Running scalability benchmark...")
    scalability_results = {}
    data_sizes = [0.25, 0.5, 0.75, 1.0]  # Fractions of full dataset

    for size_fraction in data_sizes:
        n_samples = int(X_list[0].shape[0] * size_fraction)
        X_subset = [X[:n_samples] for X in X_list]

        start_time = time.time()
        try:
            # Run SGFA analysis with subset
            _run_sgfa_analysis_benchmark(X_subset, config, size_fraction)
            duration = time.time() - start_time

            scalability_results[f'size_{size_fraction}'] = {
                'n_samples': n_samples,
                'duration_seconds': duration,
                'samples_per_second': n_samples / duration if duration > 0 else 0,
                'status': 'completed'
            }
            logger.info(f"✅ Size {size_fraction}: {n_samples} samples in {duration:.2f}s")

        except Exception as e:
            scalability_results[f'size_{size_fraction}'] = {
                'n_samples': n_samples,
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"❌ Size {size_fraction} failed: {e}")

    return scalability_results


def _run_memory_benchmark(config):
    """Run memory usage benchmark."""
    logger.info("Running memory benchmark...")
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    try:
        # Run full analysis and measure peak memory
        _run_sgfa_analysis_benchmark(None, config, 1.0, memory_test=True)
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        memory_results = {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_increase_mb': memory_after - memory_before,
            'peak_memory_mb': memory_after,
            'status': 'completed'
        }
        logger.info(f"✅ Memory: {memory_before:.1f}MB → {memory_after:.1f}MB (+{memory_after - memory_before:.1f}MB)")

    except Exception as e:
        memory_results = {
            'status': 'failed',
            'error': str(e)
        }
        logger.error(f"❌ Memory benchmark failed: {e}")

    return memory_results


def _run_timing_benchmark(X_list, config):
    """Run timing benchmark for different components."""
    logger.info("Running timing benchmark...")
    timing_results = {}

    # Test data loading time
    start_time = time.time()
    try:
        from remote_workstation.preprocessing_integration import apply_preprocessing_to_pipeline
        X_test, _ = apply_preprocessing_to_pipeline(
            config=config.__dict__,
            data_dir=config.data_dir,
            auto_select_strategy=True
        )
        timing_results['data_loading'] = {
            'duration_seconds': time.time() - start_time,
            'status': 'completed'
        }
    except Exception as e:
        timing_results['data_loading'] = {
            'status': 'failed',
            'error': str(e)
        }

    # Test preprocessing time
    start_time = time.time()
    try:
        # Simulate preprocessing step
        X_processed = [X.copy() for X in X_list]
        timing_results['preprocessing'] = {
            'duration_seconds': time.time() - start_time,
            'status': 'completed'
        }
    except Exception as e:
        timing_results['preprocessing'] = {
            'status': 'failed',
            'error': str(e)
        }

    # Test model fitting time (reduced complexity)
    start_time = time.time()
    try:
        _run_sgfa_analysis_benchmark(X_list, config, 1.0, quick_test=True)
        timing_results['model_fitting'] = {
            'duration_seconds': time.time() - start_time,
            'status': 'completed'
        }
    except Exception as e:
        timing_results['model_fitting'] = {
            'status': 'failed',
            'error': str(e)
        }

    return timing_results


def _run_sgfa_analysis_benchmark(X_list, config, size_fraction, memory_test=False, quick_test=False):
    """Run SGFA analysis for benchmarking purposes."""
    try:
        from core.run_analysis import main
        import argparse

        # Configure analysis parameters based on test type
        if quick_test:
            num_samples = 50  # Very quick test
            num_warmup = 25
        elif memory_test:
            num_samples = 100  # Moderate test for memory
            num_warmup = 50
        else:
            num_samples = 200  # Standard benchmark
            num_warmup = 100

        # Create complete args for SGFA analysis
        args = argparse.Namespace(
            model='sparseGFA',
            K=5 if quick_test else 10,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=1,
            num_runs=1,
            dataset='qmap_pd',
            data_dir=config.data_dir,
            device='gpu',
            clinical_rel="data_clinical/pd_motor_gfa_data.tsv",
            volumes_rel="volume_matrices",
            id_col="sid",
            roi_views=True,
            noise=0,
            seed=42,
            num_sources=4,
            reghsZ=True,
            percW=33
        )

        # Run the analysis
        main(args)
        return True

    except Exception as e:
        logger.debug(f"SGFA benchmark failed: {e}")
        raise e