#!/usr/bin/env python3
"""
Memory Usage Benchmarks

Benchmarks memory consumption across different operations.
"""

import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.memory_optimizer import MemoryOptimizer
from optimization.profile_utils import MemoryProfiler
from data.synthetic import generate_synthetic_data


class MemoryBenchmark:
    """Memory usage benchmark suite."""
    
    def __init__(self, output_dir="./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def benchmark_data_loading(self):
        """Benchmark memory usage during data loading."""
        print("\n" + "=" * 70)
        print("BENCHMARK: Data Loading Memory")
        print("=" * 70)
        
        configs = [
            {'n_subjects': 100, 'n_features': [500, 300]},
            {'n_subjects': 200, 'n_features': [500, 300]},
            {'n_subjects': 500, 'n_features': [500, 300]},
        ]
        
        for config in configs:
            n = config['n_subjects']
            print(f"\n  Testing n={n} subjects...")
            
            with MemoryProfiler(f"Data loading n={n}", verbose=False) as mem:
                data = generate_synthetic_data(
                    num_sources=2,
                    K=3,
                    num_subjects=n,
                    feature_sizes=config['n_features']
                )
                X_list = data['X_list']
            
            memory_usage = mem.get_memory_usage()
            
            self.results.append({
                'benchmark': 'data_loading',
                'n_subjects': n,
                'memory_mb': memory_usage.get('change_mb', 0)
            })
            
            print(f"    ✓ Memory: {memory_usage.get('change_mb', 0):.1f}MB")
        
        print("\n  Summary:")
        print(f"  {'N Subjects':<15} {'Memory (MB)':<15}")
        print("  " + "-" * 30)
        for result in [r for r in self.results if r['benchmark'] == 'data_loading']:
            print(f"  {result['n_subjects']:<15} {result['memory_mb']:<15.1f}")
    
    def benchmark_array_optimization(self):
        """Benchmark memory savings from array optimization."""
        print("\n" + "=" * 70)
        print("BENCHMARK: Array Optimization Memory Savings")
        print("=" * 70)
        
        array_sizes = [
            (100, 100),
            (500, 500),
            (1000, 1000),
        ]
        
        optimizer = MemoryOptimizer()
        
        for shape in array_sizes:
            print(f"\n  Testing array shape {shape}...")
            
            # Create float64 array
            arr = np.random.randn(*shape).astype(np.float64)
            size_before = arr.nbytes / (1024**2)  # MB
            
            # Optimize to float32
            arr_optimized = optimizer.optimize_array(arr)
            size_after = arr_optimized.nbytes / (1024**2)  # MB
            
            savings_mb = size_before - size_after
            savings_pct = (savings_mb / size_before) * 100
            
            self.results.append({
                'benchmark': 'array_optimization',
                'shape': shape,
                'size_before_mb': size_before,
                'size_after_mb': size_after,
                'savings_mb': savings_mb,
                'savings_pct': savings_pct
            })
            
            print(f"    Before: {size_before:.2f}MB, After: {size_after:.2f}MB")
            print(f"    ✓ Savings: {savings_mb:.2f}MB ({savings_pct:.1f}%)")
        
        print("\n  Summary:")
        print(f"  {'Shape':<20} {'Before (MB)':<15} {'After (MB)':<15} {'Savings (%)':<15}")
        print("  " + "-" * 65)
        for result in [r for r in self.results if r['benchmark'] == 'array_optimization']:
            print(f"  {str(result['shape']):<20} {result['size_before_mb']:<15.2f} "
                  f"{result['size_after_mb']:<15.2f} {result['savings_pct']:<15.1f}")
    
    def benchmark_preprocessing_memory(self):
        """Benchmark memory usage during preprocessing."""
        print("\n" + "=" * 70)
        print("BENCHMARK: Preprocessing Memory")
        print("=" * 70)
        
        from data.preprocessing import BasicPreprocessor
        
        configs = [
            {'n_subjects': 100, 'n_features': [200, 150]},
            {'n_subjects': 200, 'n_features': [200, 150]},
        ]
        
        for config in configs:
            n = config['n_subjects']
            print(f"\n  Testing n={n} subjects...")
            
            # Generate data
            data = generate_synthetic_data(
                num_sources=2,
                K=3,
                num_subjects=n,
                feature_sizes=config['n_features']
            )
            X_list = data['X_list']
            
            # Benchmark preprocessing
            with MemoryProfiler(f"Preprocessing n={n}", verbose=False) as mem:
                preprocessor = BasicPreprocessor()
                X_processed = preprocessor.fit_transform(
                    X_list,
                    view_names=['view1', 'view2']
                )
            
            memory_usage = mem.get_memory_usage()
            
            self.results.append({
                'benchmark': 'preprocessing',
                'n_subjects': n,
                'memory_mb': memory_usage.get('change_mb', 0)
            })
            
            print(f"    ✓ Memory: {memory_usage.get('change_mb', 0):.1f}MB")
    
    def benchmark_peak_memory_by_operation(self):
        """Benchmark peak memory by different operations."""
        print("\n" + "=" * 70)
        print("BENCHMARK: Peak Memory by Operation")
        print("=" * 70)
        
        operations = {
            'data_generation': lambda: generate_synthetic_data(2, 3, 200, [500, 300]),
            'array_creation': lambda: [np.random.randn(200, 500) for _ in range(3)],
            'matrix_multiplication': lambda: np.dot(
                np.random.randn(500, 500),
                np.random.randn(500, 500)
            ),
        }
        
        for op_name, op_func in operations.items():
            print(f"\n  Testing {op_name}...")
            
            with MemoryProfiler(op_name, verbose=False) as mem:
                result = op_func()
            
            memory_usage = mem.get_memory_usage()
            
            self.results.append({
                'benchmark': 'peak_memory',
                'operation': op_name,
                'memory_mb': memory_usage.get('change_mb', 0)
            })
            
            print(f"    ✓ Memory: {memory_usage.get('change_mb', 0):.1f}MB")
    
    def save_results(self):
        """Save benchmark results to file."""
        import json
        
        output_file = self.output_dir / "memory_benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
    
    def run_all(self):
        """Run all memory benchmarks."""
        print("\n" + "💾 " * 35)
        print("MEMORY USAGE BENCHMARKS")
        print("💾 " * 35)
        
        try:
            self.benchmark_data_loading()
        except Exception as e:
            print(f"\n⚠️  Data loading benchmark failed: {e}")
        
        try:
            self.benchmark_array_optimization()
        except Exception as e:
            print(f"\n⚠️  Array optimization benchmark failed: {e}")
        
        try:
            self.benchmark_preprocessing_memory()
        except Exception as e:
            print(f"\n⚠️  Preprocessing benchmark failed: {e}")
        
        try:
            self.benchmark_peak_memory_by_operation()
        except Exception as e:
            print(f"\n⚠️  Peak memory benchmark failed: {e}")
        
        self.save_results()
        
        print("\n" + "=" * 70)
        print("MEMORY BENCHMARKS COMPLETE")
        print("=" * 70 + "\n")


def main():
    """Run memory benchmarks."""
    benchmark = MemoryBenchmark()
    benchmark.run_all()


if __name__ == "__main__":
    main()
