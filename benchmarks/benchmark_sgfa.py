#!/usr/bin/env python3
"""
SGFA Performance Benchmarks

Benchmarks SGFA model performance across different configurations.
"""

import time
import numpy as np
import psutil
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import generate_synthetic_data
from optimization.profile_utils import SimpleTimer, MemoryProfiler


class SGFABenchmark:
    """Benchmark suite for SGFA models."""
    
    def __init__(self, output_dir="./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def benchmark_k_scaling(self):
        """Benchmark performance scaling with K (number of factors)."""
        print("\n" + "=" * 70)
        print("BENCHMARK: K Scaling")
        print("=" * 70)
        
        K_values = [2, 3, 5, 8, 10]
        n_subjects = 100
        n_features = [200, 150]
        
        for K in K_values:
            print(f"\n  Testing K={K}...")
            
            # Generate data
            data = generate_synthetic_data(
                num_sources=2,
                K=K,
                num_subjects=n_subjects,
                feature_sizes=n_features
            )
            X_list = data['X_list']
            
            # Benchmark
            with SimpleTimer(f"K={K}", verbose=False) as timer:
                with MemoryProfiler(f"K={K}", verbose=False) as mem:
                    try:
                        from core.run_analysis import run_sgfa_analysis
                        result = run_sgfa_analysis(
                            X_list,
                            K=K,
                            sparsity_level=0.25,
                            num_samples=200,  # Reduced for benchmarking
                            num_chains=2
                        )
                    except Exception as e:
                        print(f"    ✗ Failed: {e}")
                        continue
            
            duration = timer.get_duration()
            memory_change = mem.get_memory_usage()
            
            self.results.append({
                'benchmark': 'k_scaling',
                'K': K,
                'n_subjects': n_subjects,
                'duration_s': duration,
                'memory_mb': memory_change.get('change_mb', 0)
            })
            
            print(f"    ✓ Duration: {duration:.2f}s, Memory: {memory_change.get('change_mb', 0):.1f}MB")
        
        print("\n  Summary:")
        print(f"  {'K':<8} {'Time (s)':<12} {'Memory (MB)':<15}")
        print("  " + "-" * 35)
        for result in [r for r in self.results if r['benchmark'] == 'k_scaling']:
            print(f"  {result['K']:<8} {result['duration_s']:<12.2f} {result['memory_mb']:<15.1f}")
    
    def benchmark_sample_size_scaling(self):
        """Benchmark performance scaling with sample size."""
        print("\n" + "=" * 70)
        print("BENCHMARK: Sample Size Scaling")
        print("=" * 70)
        
        sample_sizes = [50, 100, 200, 500]
        K = 3
        n_features = [100, 80]
        
        for n_subjects in sample_sizes:
            print(f"\n  Testing n={n_subjects}...")
            
            # Generate data
            data = generate_synthetic_data(
                num_sources=2,
                K=K,
                num_subjects=n_subjects,
                feature_sizes=n_features
            )
            X_list = data['X_list']
            
            # Benchmark
            with SimpleTimer(f"n={n_subjects}", verbose=False) as timer:
                with MemoryProfiler(f"n={n_subjects}", verbose=False) as mem:
                    try:
                        from core.run_analysis import run_sgfa_analysis
                        result = run_sgfa_analysis(
                            X_list,
                            K=K,
                            sparsity_level=0.25,
                            num_samples=200,
                            num_chains=2
                        )
                    except Exception as e:
                        print(f"    ✗ Failed: {e}")
                        continue
            
            duration = timer.get_duration()
            memory_change = mem.get_memory_usage()
            
            self.results.append({
                'benchmark': 'sample_scaling',
                'n_subjects': n_subjects,
                'K': K,
                'duration_s': duration,
                'memory_mb': memory_change.get('change_mb', 0)
            })
            
            print(f"    ✓ Duration: {duration:.2f}s, Memory: {memory_change.get('change_mb', 0):.1f}MB")
        
        print("\n  Summary:")
        print(f"  {'N Subjects':<12} {'Time (s)':<12} {'Memory (MB)':<15}")
        print("  " + "-" * 39)
        for result in [r for r in self.results if r['benchmark'] == 'sample_scaling']:
            print(f"  {result['n_subjects']:<12} {result['duration_s']:<12.2f} {result['memory_mb']:<15.1f}")
    
    def benchmark_sparsity_levels(self):
        """Benchmark performance across different sparsity levels."""
        print("\n" + "=" * 70)
        print("BENCHMARK: Sparsity Levels")
        print("=" * 70)
        
        sparsity_levels = [0.1, 0.25, 0.33, 0.5]
        K = 3
        n_subjects = 100
        n_features = [100, 80]
        
        for sparsity in sparsity_levels:
            print(f"\n  Testing sparsity={sparsity}...")
            
            # Generate data
            data = generate_synthetic_data(
                num_sources=2,
                K=K,
                num_subjects=n_subjects,
                feature_sizes=n_features
            )
            X_list = data['X_list']
            
            # Benchmark
            with SimpleTimer(f"sparsity={sparsity}", verbose=False) as timer:
                try:
                    from core.run_analysis import run_sgfa_analysis
                    result = run_sgfa_analysis(
                        X_list,
                        K=K,
                        sparsity_level=sparsity,
                        num_samples=200,
                        num_chains=2
                    )
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
                    continue
            
            duration = timer.get_duration()
            
            self.results.append({
                'benchmark': 'sparsity',
                'sparsity': sparsity,
                'K': K,
                'n_subjects': n_subjects,
                'duration_s': duration
            })
            
            print(f"    ✓ Duration: {duration:.2f}s")
        
        print("\n  Summary:")
        print(f"  {'Sparsity':<12} {'Time (s)':<12}")
        print("  " + "-" * 24)
        for result in [r for r in self.results if r['benchmark'] == 'sparsity']:
            print(f"  {result['sparsity']:<12.2f} {result['duration_s']:<12.2f}")
    
    def benchmark_mcmc_parameters(self):
        """Benchmark performance across MCMC parameter settings."""
        print("\n" + "=" * 70)
        print("BENCHMARK: MCMC Parameters")
        print("=" * 70)
        
        mcmc_configs = [
            {'samples': 100, 'chains': 2},
            {'samples': 200, 'chains': 2},
            {'samples': 500, 'chains': 2},
            {'samples': 200, 'chains': 4},
        ]
        
        K = 3
        n_subjects = 100
        n_features = [100, 80]
        
        # Generate data once
        data = generate_synthetic_data(
            num_sources=2,
            K=K,
            num_subjects=n_subjects,
            feature_sizes=n_features
        )
        X_list = data['X_list']
        
        for config in mcmc_configs:
            print(f"\n  Testing samples={config['samples']}, chains={config['chains']}...")
            
            # Benchmark
            with SimpleTimer(f"MCMC config", verbose=False) as timer:
                try:
                    from core.run_analysis import run_sgfa_analysis
                    result = run_sgfa_analysis(
                        X_list,
                        K=K,
                        sparsity_level=0.25,
                        num_samples=config['samples'],
                        num_chains=config['chains']
                    )
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
                    continue
            
            duration = timer.get_duration()
            
            self.results.append({
                'benchmark': 'mcmc',
                'samples': config['samples'],
                'chains': config['chains'],
                'duration_s': duration
            })
            
            print(f"    ✓ Duration: {duration:.2f}s")
        
        print("\n  Summary:")
        print(f"  {'Samples':<10} {'Chains':<10} {'Time (s)':<12}")
        print("  " + "-" * 32)
        for result in [r for r in self.results if r['benchmark'] == 'mcmc']:
            print(f"  {result['samples']:<10} {result['chains']:<10} {result['duration_s']:<12.2f}")
    
    def save_results(self):
        """Save benchmark results to file."""
        import json
        
        output_file = self.output_dir / "benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
    
    def run_all(self):
        """Run all benchmarks."""
        print("\n" + "🔥 " * 35)
        print("SGFA PERFORMANCE BENCHMARKS")
        print("🔥 " * 35)
        
        try:
            self.benchmark_k_scaling()
        except Exception as e:
            print(f"\n⚠️  K scaling benchmark failed: {e}")
        
        try:
            self.benchmark_sample_size_scaling()
        except Exception as e:
            print(f"\n⚠️  Sample size benchmark failed: {e}")
        
        try:
            self.benchmark_sparsity_levels()
        except Exception as e:
            print(f"\n⚠️  Sparsity benchmark failed: {e}")
        
        try:
            self.benchmark_mcmc_parameters()
        except Exception as e:
            print(f"\n⚠️  MCMC benchmark failed: {e}")
        
        self.save_results()
        
        print("\n" + "=" * 70)
        print("BENCHMARKS COMPLETE")
        print("=" * 70 + "\n")


def main():
    """Run benchmarks."""
    benchmark = SGFABenchmark()
    benchmark.run_all()


if __name__ == "__main__":
    main()
