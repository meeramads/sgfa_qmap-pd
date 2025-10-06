# SGFA qMAP-PD Benchmarking Suite

Performance and memory benchmarks for the SGFA qMAP-PD framework.

## Overview

This directory contains benchmarking scripts to measure:
- **Performance**: Execution time across configurations
- **Memory**: Memory consumption and optimization
- **Scalability**: How performance scales with data size

## Benchmark Scripts

### 1. SGFA Performance Benchmarks
**[benchmark_sgfa.py](benchmark_sgfa.py)** - Core SGFA performance benchmarks

**Tests:**
- K scaling (number of factors)
- Sample size scaling
- Sparsity level performance
- MCMC parameter effects

**Usage:**
```bash
python benchmarks/benchmark_sgfa.py
```

**Output:**
- Console output with timing results
- `benchmark_results/benchmark_results.json`

### 2. Memory Benchmarks
**[benchmark_memory.py](benchmark_memory.py)** - Memory usage benchmarks

**Tests:**
- Data loading memory
- Array optimization savings
- Preprocessing memory
- Peak memory by operation

**Usage:**
```bash
python benchmarks/benchmark_memory.py
```

**Output:**
- Console output with memory usage
- `benchmark_results/memory_benchmark_results.json`

## Quick Start

### Run All Benchmarks

```bash
# SGFA performance
python benchmarks/benchmark_sgfa.py

# Memory usage
python benchmarks/benchmark_memory.py
```

### Run Specific Benchmark

```python
from benchmarks.benchmark_sgfa import SGFABenchmark

benchmark = SGFABenchmark()
benchmark.benchmark_k_scaling()
benchmark.save_results()
```

## Benchmark Results

Results are saved to `benchmark_results/` in JSON format:

```json
{
  "benchmark": "k_scaling",
  "K": 5,
  "n_subjects": 100,
  "duration_s": 45.23,
  "memory_mb": 512.5
}
```

## Performance Expectations

### Typical Performance (CPU mode)

**K Scaling (n=100, 2 views):**
- K=2: ~20-30s
- K=3: ~30-40s
- K=5: ~40-60s
- K=8: ~60-90s

**Sample Size Scaling (K=3, 2 views):**
- n=50: ~15-20s
- n=100: ~30-40s
- n=200: ~50-70s
- n=500: ~100-150s

### Memory Usage

**Data Loading:**
- 100 subjects: ~50-100 MB
- 200 subjects: ~100-200 MB
- 500 subjects: ~250-500 MB

**Array Optimization:**
- float64 → float32: ~50% memory savings

## Interpreting Results

### Performance Metrics

1. **Duration (seconds)**: Wall-clock time to complete operation
2. **Memory (MB)**: Peak memory usage or change during operation

### Scaling Patterns

- **Linear scaling**: Performance increases proportionally with data size
- **Superlinear scaling**: Performance increases faster than data size (suboptimal)
- **Sublinear scaling**: Performance increases slower than data size (optimal)

### Performance Tips

1. **Use GPU**: Significant speedup (5-10x) when available
2. **Optimize K**: Use minimum K needed for your analysis
3. **Reduce MCMC samples**: For development, use fewer samples
4. **Enable optimization**: Use memory optimizer and data streaming
5. **Batch processing**: Process large datasets in chunks

## Customizing Benchmarks

### Add New Benchmark

```python
class MyBenchmark(SGFABenchmark):
    def benchmark_my_operation(self):
        """Benchmark my custom operation."""
        from optimization.profile_utils import SimpleTimer
        
        with SimpleTimer("My operation") as timer:
            # Your operation here
            result = my_custom_operation()
        
        self.results.append({
            'benchmark': 'my_operation',
            'duration_s': timer.get_duration()
        })
```

### Modify Configuration

Edit the benchmark scripts to test different:
- Data sizes
- K values
- MCMC parameters
- Sparsity levels

## Continuous Benchmarking

### Track Performance Over Time

```bash
# Run benchmarks and save with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python benchmarks/benchmark_sgfa.py
mv benchmark_results/benchmark_results.json \
   benchmark_results/benchmark_${TIMESTAMP}.json
```

### Compare Results

```python
import json
import pandas as pd

# Load multiple benchmark results
results = []
for file in Path('benchmark_results').glob('benchmark_*.json'):
    with open(file) as f:
        data = json.load(f)
        results.extend(data)

# Create DataFrame for analysis
df = pd.DataFrame(results)
print(df.groupby('K')['duration_s'].mean())
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Run performance benchmarks
  run: |
    python benchmarks/benchmark_sgfa.py
    python benchmarks/benchmark_memory.py

- name: Upload benchmark results
  uses: actions/upload-artifact@v3
  with:
    name: benchmarks
    path: benchmark_results/
```

## Profiling Tools

For detailed profiling beyond benchmarks:

### CPU Profiling
```bash
python -m cProfile -o profile.stats benchmarks/benchmark_sgfa.py
python -m pstats profile.stats
```

### Memory Profiling
```bash
pip install memory-profiler
python -m memory_profiler benchmarks/benchmark_memory.py
```

### Line-by-Line Profiling
```bash
pip install line-profiler
kernprof -l -v benchmarks/benchmark_sgfa.py
```

## Troubleshooting

### Benchmarks Run Slowly

1. Reduce MCMC samples in benchmark configs
2. Use smaller data sizes for quick tests
3. Enable GPU acceleration
4. Close other applications

### Out of Memory Errors

1. Reduce data sizes in benchmarks
2. Enable memory optimization
3. Use data streaming for large datasets
4. Increase system swap space

### Inconsistent Results

1. Run benchmarks multiple times and average
2. Close background applications
3. Use fixed random seeds
4. Monitor system load during benchmarks

## Best Practices

1. **Warm-up**: Run once before timing to warm caches
2. **Multiple runs**: Average over 3-5 runs for stability
3. **Isolated environment**: Minimize background processes
4. **Consistent hardware**: Use same machine for comparisons
5. **Document changes**: Note code changes between benchmark runs

## See Also

- [optimization/PROFILING_GUIDE.md](../optimization/PROFILING_GUIDE.md) - Detailed profiling guide
- [optimization/profile_utils.py](../optimization/profile_utils.py) - Profiling utilities
- [tests/optimization/](../tests/optimization/) - Optimization tests

---

**Ready to benchmark!** 🚀📊
