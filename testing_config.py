"""
Comprehensive Testing Configuration for Sparse Bayesian Group Factor Analysis.

This module provides:
1. Smoke tests for quick validation
2. Synthetic data demonstrations
3. qMAP-PD experiments with/without preprocessing and CV
4. Hyperparameter tuning configurations
5. Benchmark tests for performance evaluation
"""

import logging
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestConfiguration:
    """Base configuration class for different test types."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.base_args = {}
        self.expected_runtime_minutes = 1
        self.memory_gb = 1
        self.gpu_recommended = False
        
    def get_args(self) -> Dict:
        """Get command line arguments as dictionary."""
        return self.base_args.copy()
    
    def get_cmd_string(self) -> str:
        """Get command line string for running the test."""
        args_str = " ".join([f"--{k.replace('_', '-')} {v}" for k, v in self.base_args.items() if v is not True])
        flags_str = " ".join([f"--{k.replace('_', '-')}" for k, v in self.base_args.items() if v is True])
        return f"python run_analysis.py {args_str} {flags_str}".strip()

class SmokeTestConfig(TestConfiguration):
    """Minimal tests for quick validation that the pipeline works."""
    
    def __init__(self, test_id: str):
        super().__init__(f"smoke_{test_id}", "Smoke test for quick validation")
        self.expected_runtime_minutes = 2
        self.memory_gb = 1

class SyntheticTestConfig(TestConfiguration):
    """Tests with synthetic data to demonstrate model behavior."""
    
    def __init__(self, test_id: str):
        super().__init__(f"synthetic_{test_id}", "Synthetic data demonstration")
        self.expected_runtime_minutes = 5
        self.memory_gb = 2

class QMAPTestConfig(TestConfiguration):
    """Tests with qMAP-PD data for real-world validation."""
    
    def __init__(self, test_id: str):
        super().__init__(f"qmap_{test_id}", "qMAP-PD real data analysis")
        self.expected_runtime_minutes = 15
        self.memory_gb = 4
        self.gpu_recommended = True

class HyperparameterTestConfig(TestConfiguration):
    """Tests for hyperparameter tuning and optimization."""
    
    def __init__(self, test_id: str):
        super().__init__(f"hyperparam_{test_id}", "Hyperparameter tuning experiment")
        self.expected_runtime_minutes = 60
        self.memory_gb = 8
        self.gpu_recommended = True

# == SMOKE TESTS ==

def get_smoke_tests() -> List[TestConfiguration]:
    """Get minimal smoke tests for quick validation."""
    
    tests = []
    
    # Test 1: Basic synthetic data
    test = SmokeTestConfig("basic_synthetic")
    test.description = "Basic synthetic data with minimal parameters"
    test.base_args = {
        'dataset': 'synthetic',
        'K': 3,
        'num_samples': 100,
        'num_warmup': 50,
        'num_chains': 1,
        'num_runs': 1,
        'percW': 33,
        'device': 'cpu',
        'seed': 42
    }
    tests.append(test)
    
    # Test 2: Basic qMAP-PD (if data available)
    test = SmokeTestConfig("basic_qmap")
    test.description = "Basic qMAP-PD with minimal parameters"
    test.base_args = {
        'dataset': 'qmap_pd',
        'K': 5,
        'num_samples': 200,
        'num_warmup': 100,
        'num_chains': 1,
        'num_runs': 1,
        'percW': 33,
        'device': 'cpu',
        'seed': 42
    }
    tests.append(test)
    
    # Test 3: Preprocessing pipeline
    test = SmokeTestConfig("preprocessing")
    test.description = "Test preprocessing pipeline"
    test.base_args = {
        'dataset': 'qmap_pd',
        'K': 3,
        'num_samples': 100,
        'num_warmup': 50,
        'num_chains': 1,
        'num_runs': 1,
        'enable_preprocessing': True,
        'feature_selection': 'variance',
        'n_top_features': 100,
        'seed': 42
    }
    tests.append(test)
    
    # Test 4: Cross-validation smoke test
    test = SmokeTestConfig("cv_basic")
    test.description = "Basic cross-validation test"
    test.base_args = {
        'dataset': 'synthetic',
        'K': 3,
        'num_samples': 50,
        'num_warmup': 25,
        'num_chains': 1,
        'cv_only': True,
        'cv_folds': 3,
        'seed': 42
    }
    tests.append(test)
    
    return tests

# == SYNTHETIC DATA DEMONSTRATIONS ==

def get_synthetic_demonstrations() -> List[TestConfiguration]:
    """Get synthetic data tests to demonstrate model behavior."""
    
    tests = []
    
    # Demo 1: Factor recovery with different sparsity levels
    for sparsity in [25, 50, 75]:
        test = SyntheticTestConfig(f"sparsity_{sparsity}")
        test.description = f"Factor recovery with {sparsity}% sparsity"
        test.base_args = {
            'dataset': 'synthetic',
            'K': 5,
            'num_samples': 1000,
            'num_warmup': 500,
            'num_chains': 2,
            'num_runs': 3,
            'percW': sparsity,
            'device': 'cpu',
            'seed': 42
        }
        tests.append(test)
    
    # Demo 2: Varying number of factors
    for K in [3, 5, 10]:
        test = SyntheticTestConfig(f"factors_{K}")
        test.description = f"Model behavior with {K} latent factors"
        test.base_args = {
            'dataset': 'synthetic',
            'K': K,
            'num_samples': 1500,
            'num_warmup': 750,
            'num_chains': 2,
            'num_runs': 3,
            'percW': 33,
            'device': 'cpu',
            'seed': 42
        }
        tests.append(test)
    
    # Demo 3: Regularized vs non-regularized horseshoe
    for reghsZ in [True, False]:
        test = SyntheticTestConfig(f"reghsZ_{reghsZ}")
        test.description = f"Regularized horseshoe Z: {reghsZ}"
        test.base_args = {
            'dataset': 'synthetic',
            'K': 5,
            'num_samples': 1000,
            'num_warmup': 500,
            'num_chains': 2,
            'num_runs': 2,
            'percW': 33,
            'reghsZ': reghsZ,
            'device': 'cpu',
            'seed': 42
        }
        tests.append(test)
    
    # Demo 4: Model comparison (sparse vs standard GFA)
    for model in ['sparseGFA', 'GFA']:
        test = SyntheticTestConfig(f"model_{model}")
        test.description = f"Demonstrate {model} model"
        test.base_args = {
            'dataset': 'synthetic',
            'model': model,
            'K': 5,
            'num_samples': 1000,
            'num_warmup': 500,
            'num_chains': 2,
            'num_runs': 3,
            'percW': 33,
            'device': 'cpu',
            'seed': 42
        }
        tests.append(test)
    
    # Demo 5: Cross-validation on synthetic data
    test = SyntheticTestConfig("cv_demo")
    test.description = "Cross-validation demonstration with synthetic data"
    test.base_args = {
        'dataset': 'synthetic',
        'K': 5,
        'num_samples': 800,
        'num_warmup': 400,
        'num_chains': 1,
        'run_cv': True,
        'cv_folds': 5,
        'seed': 42
    }
    tests.append(test)
    
    return tests

# == qMAP-PD EXPERIMENTS ==

def get_qmap_experiments() -> List[TestConfiguration]:
    """Get qMAP-PD experiments with different configurations."""
    
    tests = []
    
    # Experiment 1: Basic qMAP-PD analysis
    test = QMAPTestConfig("basic")
    test.description = "Standard qMAP-PD analysis without preprocessing"
    test.base_args = {
        'dataset': 'qmap_pd',
        'K': 20,
        'num_samples': 3000,
        'num_warmup': 1500,
        'num_chains': 4,
        'num_runs': 5,
        'percW': 33,
        'device': 'gpu',
        'seed': 42
    }
    test.expected_runtime_minutes = 30
    tests.append(test)
    
    # Experiment 2: qMAP-PD with preprocessing
    test = QMAPTestConfig("with_preprocessing")
    test.description = "qMAP-PD with advanced preprocessing"
    test.base_args = {
        'dataset': 'qmap_pd',
        'K': 20,
        'num_samples': 3000,
        'num_warmup': 1500,
        'num_chains': 4,
        'num_runs': 5,
        'percW': 33,
        'enable_preprocessing': True,
        'feature_selection': 'statistical',
        'n_top_features': 500,
        'imputation_strategy': 'knn',
        'device': 'gpu',
        'seed': 42
    }
    test.expected_runtime_minutes = 35
    tests.append(test)
    
    # Experiment 3: qMAP-PD with cross-validation
    test = QMAPTestConfig("with_cv")
    test.description = "qMAP-PD with cross-validation"
    test.base_args = {
        'dataset': 'qmap_pd',
        'K': 15,
        'num_samples': 2000,
        'num_warmup': 1000,
        'num_chains': 2,
        'run_cv': True,
        'cv_folds': 5,
        'cv_type': 'standard',
        'device': 'gpu',
        'seed': 42
    }
    test.expected_runtime_minutes = 45
    tests.append(test)
    
    # Experiment 4: Full pipeline (preprocessing + CV)
    test = QMAPTestConfig("full_pipeline")
    test.description = "Complete pipeline with preprocessing and CV"
    test.base_args = {
        'dataset': 'qmap_pd',
        'K': 20,
        'num_samples': 2500,
        'num_warmup': 1250,
        'num_chains': 3,
        'enable_preprocessing': True,
        'feature_selection': 'combined',
        'n_top_features': 400,
        'imputation_strategy': 'knn',
        'run_cv': True,
        'cv_folds': 5,
        'device': 'gpu',
        'seed': 42
    }
    test.expected_runtime_minutes = 60
    test.memory_gb = 6
    tests.append(test)
    
    # Experiment 5: ROI-separated analysis
    test = QMAPTestConfig("roi_separated")
    test.description = "Analysis with separate ROI views"
    test.base_args = {
        'dataset': 'qmap_pd',
        'roi_views': True,
        'K': 25,
        'num_samples': 3000,
        'num_warmup': 1500,
        'num_chains': 4,
        'num_runs': 3,
        'percW': 33,
        'device': 'gpu',
        'seed': 42
    }
    test.expected_runtime_minutes = 35
    tests.append(test)
    
    # Experiment 6: Factor-to-MRI mapping
    test = QMAPTestConfig("factor_mapping")
    test.description = "Analysis with factor-to-MRI mapping"
    test.base_args = {
        'dataset': 'qmap_pd',
        'K': 15,
        'num_samples': 2000,
        'num_warmup': 1000,
        'num_chains': 2,
        'num_runs': 3,
        'create_factor_maps': True,
        'device': 'gpu',
        'seed': 42
    }
    test.expected_runtime_minutes = 25
    tests.append(test)
    
    return tests

# == HYPERPARAMETER TUNING EXPERIMENTS ==

def get_hyperparameter_experiments() -> List[TestConfiguration]:
    """Get hyperparameter tuning experiments."""
    
    tests = []
    
    # HPT 1: K (number of factors) tuning
    test = HyperparameterTestConfig("K_tuning")
    test.description = "Hyperparameter tuning for optimal number of factors"
    test.base_args = {
        'dataset': 'qmap_pd',
        'nested_cv': True,
        'cv_folds': 5,
        'K': 20,  # Starting point
        'num_samples': 1500,
        'num_warmup': 750,
        'num_chains': 2,
        'device': 'gpu',
        'seed': 42
    }
    test.expected_runtime_minutes = 90
    test.memory_gb = 8
    tests.append(test)
    
    # HPT 2: Sparsity level tuning
    test = HyperparameterTestConfig("sparsity_tuning")
    test.description = "Tuning sparsity percentage"
    test.base_args = {
        'dataset': 'qmap_pd',
        'nested_cv': True,
        'cv_folds': 5,
        'K': 15,
        'percW': 33,  # Starting point
        'num_samples': 1500,
        'num_warmup': 750,
        'num_chains': 2,
        'device': 'gpu',
        'seed': 42
    }
    test.expected_runtime_minutes = 75
    tests.append(test)
    
    # HPT 3: Preprocessing parameter tuning
    test = HyperparameterTestConfig("preprocessing_tuning")
    test.description = "Optimize preprocessing parameters"
    test.base_args = {
        'dataset': 'qmap_pd',
        'K': 20,
        'num_samples': 1000,
        'num_warmup': 500,
        'num_chains': 2,
        'enable_preprocessing': True,
        'optimize_preprocessing': True,
        'cross_validate_sources': True,
        'device': 'gpu',
        'seed': 42
    }
    test.expected_runtime_minutes = 45
    tests.append(test)
    
    # HPT 4: Complete hyperparameter grid search
    test = HyperparameterTestConfig("grid_search")
    test.description = "Comprehensive grid search over multiple parameters"
    test.base_args = {
        'dataset': 'qmap_pd',
        'nested_cv': True,
        'cv_folds': 7,
        'K': 20,
        'num_samples': 2000,
        'num_warmup': 1000,
        'num_chains': 3,
        'enable_preprocessing': True,
        'feature_selection': 'combined',
        'device': 'gpu',
        'seed': 42
    }
    test.expected_runtime_minutes = 120
    test.memory_gb = 12
    tests.append(test)
    
    return tests

# == CONFIGURATION MANAGER ==

class TestConfigManager:
    """Manages and organizes all test configurations."""
    
    def __init__(self):
        self.smoke_tests = get_smoke_tests()
        self.synthetic_demos = get_synthetic_demonstrations()
        self.qmap_experiments = get_qmap_experiments()
        self.hyperparam_experiments = get_hyperparameter_experiments()
        
        self.all_tests = (
            self.smoke_tests + 
            self.synthetic_demos + 
            self.qmap_experiments + 
            self.hyperparam_experiments
        )
    
    def get_test_by_name(self, name: str) -> Optional[TestConfiguration]:
        """Get test configuration by name."""
        for test in self.all_tests:
            if test.name == name:
                return test
        return None
    
    def get_tests_by_category(self, category: str) -> List[TestConfiguration]:
        """Get tests by category."""
        category_map = {
            'smoke': self.smoke_tests,
            'synthetic': self.synthetic_demos,
            'qmap': self.qmap_experiments,
            'hyperparam': self.hyperparam_experiments
        }
        return category_map.get(category, [])
    
    def list_all_tests(self) -> None:
        """Print all available tests."""
        print("Available Test Configurations:")
        print("=" * 50)
        
        categories = [
            ("Smoke Tests", self.smoke_tests),
            ("Synthetic Demonstrations", self.synthetic_demos),
            ("qMAP-PD Experiments", self.qmap_experiments),
            ("Hyperparameter Tuning", self.hyperparam_experiments)
        ]
        
        for cat_name, tests in categories:
            print(f"\n{cat_name}:")
            print("-" * len(cat_name))
            for test in tests:
                runtime_str = f"{test.expected_runtime_minutes}min"
                memory_str = f"{test.memory_gb}GB"
                gpu_str = " (GPU rec.)" if test.gpu_recommended else ""
                print(f"  {test.name:20s} - {test.description} [{runtime_str}, {memory_str}{gpu_str}]")
    
    def generate_batch_script(self, test_names: List[str], output_file: str = "run_tests.sh"):
        """Generate batch script to run multiple tests."""
        
        script_content = """#!/bin/bash

# Batch test execution script
# Generated automatically by TestConfigManager

set -e  # Exit on error

echo "Starting batch test execution..."
echo "======================================"

"""
        
        total_time = 0
        for test_name in test_names:
            test = self.get_test_by_name(test_name)
            if test:
                script_content += f"""
echo "Running test: {test.name}"
echo "Description: {test.description}"
echo "Expected runtime: {test.expected_runtime_minutes} minutes"
echo "--------------------------------------"

{test.get_cmd_string()}

if [ $? -eq 0 ]; then
    echo "✓ Test {test.name} completed successfully"
else
    echo "✗ Test {test.name} failed"
    exit 1
fi

echo ""
"""
                total_time += test.expected_runtime_minutes
        
        script_content += f"""
echo "======================================"
echo "All tests completed successfully!"
echo "Total estimated runtime: {total_time} minutes"
"""
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        # Make executable
        import stat
        st = os.stat(output_file)
        os.chmod(output_file, st.st_mode | stat.S_IEXEC)
        
        print(f"Batch script saved to: {output_file}")
        print(f"Total estimated runtime: {total_time} minutes")
        
        return output_file
    
    def save_test_configs(self, output_file: str = "test_configurations.json"):
        """Save all test configurations to JSON for reference."""
        
        configs = {}
        
        for test in self.all_tests:
            configs[test.name] = {
                'description': test.description,
                'args': test.base_args,
                'expected_runtime_minutes': test.expected_runtime_minutes,
                'memory_gb': test.memory_gb,
                'gpu_recommended': test.gpu_recommended,
                'command': test.get_cmd_string()
            }
        
        with open(output_file, 'w') as f:
            json.dump(configs, f, indent=2)
        
        print(f"Test configurations saved to: {output_file}")
        
        return output_file

# == BENCHMARK AND VALIDATION TOOLS ==

class TestRunner:
    """Utility class for running and validating tests."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.results = {}
    
    def run_test(self, test_config: TestConfiguration, timeout_multiplier: float = 2.0) -> Dict:
        """
        Run a single test configuration.
        
        Parameters:
        -----------
        test_config : TestConfiguration
            Test to run
        timeout_multiplier : float
            Multiply expected runtime for timeout (default 2x)
        
        Returns:
        --------
        dict : Test results including success status, runtime, etc.
        """
        
        result = {
            'test_name': test_config.name,
            'description': test_config.description,
            'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'success': False,
            'runtime_minutes': 0,
            'error': None,
            'command': test_config.get_cmd_string()
        }
        
        if self.dry_run:
            print(f"DRY RUN: Would execute: {test_config.get_cmd_string()}")
            result['success'] = True
            result['runtime_minutes'] = test_config.expected_runtime_minutes
            return result
        
        print(f"Running test: {test_config.name}")
        print(f"Command: {test_config.get_cmd_string()}")
        print(f"Expected runtime: {test_config.expected_runtime_minutes} minutes")
        
        start_time = time.time()
        timeout_seconds = test_config.expected_runtime_minutes * 60 * timeout_multiplier
        
        try:
            import subprocess
            
            # Build command
            cmd_parts = test_config.get_cmd_string().split()
            
            # Run with timeout
            process = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            try:
                stdout, _ = process.communicate(timeout=timeout_seconds)
                
                if process.returncode == 0:
                    result['success'] = True
                    print(f"✓ Test {test_config.name} completed successfully")
                else:
                    result['error'] = f"Process exited with code {process.returncode}"
                    print(f"✗ Test {test_config.name} failed")
                
            except subprocess.TimeoutExpired:
                process.kill()
                result['error'] = f"Test timed out after {timeout_seconds/60:.1f} minutes"
                print(f"✗ Test {test_config.name} timed out")
        
        except Exception as e:
            result['error'] = str(e)
            print(f"✗ Test {test_config.name} failed with error: {e}")
        
        result['runtime_minutes'] = (time.time() - start_time) / 60
        result['completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        self.results[test_config.name] = result
        return result
    
    def run_test_suite(self, test_names: List[str]) -> Dict:
        """Run multiple tests and return summary results."""
        
        print(f"Running test suite with {len(test_names)} tests...")
        print("=" * 60)
        
        manager = TestConfigManager()
        
        suite_results = {
            'started_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_results': {},
            'summary': {
                'total_tests': len(test_names),
                'successful_tests': 0,
                'failed_tests': 0,
                'total_runtime_minutes': 0
            }
        }
        
        for test_name in test_names:
            test_config = manager.get_test_by_name(test_name)
            if test_config is None:
                print(f"Warning: Test '{test_name}' not found, skipping...")
                continue
            
            result = self.run_test(test_config)
            suite_results['test_results'][test_name] = result
            suite_results['summary']['total_runtime_minutes'] += result['runtime_minutes']
            
            if result['success']:
                suite_results['summary']['successful_tests'] += 1
            else:
                suite_results['summary']['failed_tests'] += 1
        
        suite_results['completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUITE SUMMARY")
        print("=" * 60)
        summary = suite_results['summary']
        print(f"Total tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success rate: {summary['successful_tests']/summary['total_tests']*100:.1f}%")
        print(f"Total runtime: {summary['total_runtime_minutes']:.1f} minutes")
        
        return suite_results
    
    def save_results(self, output_file: str = "test_results.json"):
        """Save test results to file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Test results saved to: {output_file}")

# == PREDEFINED TEST SUITES ==

def get_quick_validation_suite() -> List[str]:
    """Get test names for quick validation (all smoke tests)."""
    return [test.name for test in get_smoke_tests()]

def get_demonstration_suite() -> List[str]:
    """Get test names for comprehensive demonstration."""
    tests = []
    tests.extend([test.name for test in get_smoke_tests()])
    tests.extend([test.name for test in get_synthetic_demonstrations()[:3]])  # First 3 synthetic
    tests.append('qmap_basic')  # One qMAP-PD test
    return tests

def get_full_evaluation_suite() -> List[str]:
    """Get test names for full model evaluation."""
    tests = []
    tests.extend([test.name for test in get_smoke_tests()])
    tests.extend([test.name for test in get_synthetic_demonstrations()])
    tests.extend([test.name for test in get_qmap_experiments()])
    return tests

def get_hyperparameter_suite() -> List[str]:
    """Get test names for hyperparameter optimization."""
    return [test.name for test in get_hyperparameter_experiments()]

# == CLI FUNCTIONALITY ==

def main():
    """Main CLI function for test configuration."""
    
    parser = argparse.ArgumentParser(description="Test Configuration Manager for Sparse GFA")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List tests command
    list_parser = subparsers.add_parser('list', help='List all available tests')
    list_parser.add_argument('--category', choices=['smoke', 'synthetic', 'qmap', 'hyperparam'],
                            help='Filter by test category')
    
    # Generate script command
    script_parser = subparsers.add_parser('generate-script', help='Generate batch execution script')
    script_parser.add_argument('--tests', nargs='+', help='Test names to include')
    script_parser.add_argument('--suite', choices=['quick', 'demo', 'full', 'hyperparam'],
                              help='Use predefined test suite')
    script_parser.add_argument('--output', default='run_tests.sh', help='Output script filename')
    
    # Run tests command
    run_parser = subparsers.add_parser('run', help='Run tests directly')
    run_parser.add_argument('--tests', nargs='+', help='Test names to run')
    run_parser.add_argument('--suite', choices=['quick', 'demo', 'full', 'hyperparam'],
                           help='Use predefined test suite')
    run_parser.add_argument('--dry-run', action='store_true', help='Show commands without executing')
    run_parser.add_argument('--save-results', default='test_results.json', help='Save results to file')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get info about specific test')
    info_parser.add_argument('test_name', help='Test name to get info about')
    
    # Export configs command
    export_parser = subparsers.add_parser('export', help='Export test configurations to JSON')
    export_parser.add_argument('--output', default='test_configurations.json', help='Output JSON filename')
    
    args = parser.parse_args()
    
    manager = TestConfigManager()
    
    if args.command == 'list':
        if args.category:
            tests = manager.get_tests_by_category(args.category)
            print(f"{args.category.title()} Tests:")
            print("-" * 30)
            for test in tests:
                print(f"  {test.name:20s} - {test.description}")
        else:
            manager.list_all_tests()
    
    elif args.command == 'generate-script':
        # Determine test names
        if args.suite:
            suite_map = {
                'quick': get_quick_validation_suite(),
                'demo': get_demonstration_suite(),
                'full': get_full_evaluation_suite(),
                'hyperparam': get_hyperparameter_suite()
            }
            test_names = suite_map[args.suite]
        elif args.tests:
            test_names = args.tests
        else:
            print("Error: Must specify either --tests or --suite")
            return
        
        manager.generate_batch_script(test_names, args.output)
    
    elif args.command == 'run':
        # Determine test names
        if args.suite:
            suite_map = {
                'quick': get_quick_validation_suite(),
                'demo': get_demonstration_suite(),
                'full': get_full_evaluation_suite(),
                'hyperparam': get_hyperparameter_suite()
            }
            test_names = suite_map[args.suite]
        elif args.tests:
            test_names = args.tests
        else:
            print("Error: Must specify either --tests or --suite")
            return
        
        # Run tests
        runner = TestRunner(dry_run=args.dry_run)
        results = runner.run_test_suite(test_names)
        
        if args.save_results:
            runner.save_results(args.save_results)
    
    elif args.command == 'info':
        test = manager.get_test_by_name(args.test_name)
        if test:
            print(f"Test: {test.name}")
            print(f"Description: {test.description}")
            print(f"Expected runtime: {test.expected_runtime_minutes} minutes")
            print(f"Memory requirement: {test.memory_gb} GB")
            print(f"GPU recommended: {test.gpu_recommended}")
            print(f"Command: {test.get_cmd_string()}")
        else:
            print(f"Test '{args.test_name}' not found")
            manager.list_all_tests()
    
    elif args.command == 'export':
        manager.save_test_configs(args.output)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
