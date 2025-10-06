# Profiling Guide

Simple guide for profiling SGFA qMAP-PD code performance.

## Quick Start

### Time a Function

```python
from optimization.profile_utils import time_function

@time_function
def my_analysis():
    # Your code here
    pass
```

### Profile a Function

```python
from optimization.profile_utils import profile_function

@profile_function(output_file='results/profile.txt', top_n=20)
def my_analysis():
    # Your code here
    pass
```

### Time a Code Block

```python
from optimization.profile_utils import SimpleTimer

with SimpleTimer("Data loading"):
    X_list = load_data()

with SimpleTimer("Model training"):
    results = train_model(X_list)
```

### Track Memory Usage

```python
from optimization.profile_utils import MemoryProfiler

with MemoryProfiler("SGFA training"):
    results = run_sgfa_analysis(X_list)
```

### Quick Profile (All-in-One)

```python
from optimization.profile_utils import quick_profile

@quick_profile(time_it=True, memory_it=True)
def my_experiment():
    # Your code here
    pass
```

## Advanced Usage

### Profile Experiment Scripts

Add profiling to your experiment scripts:

```python
from optimization.profile_utils import SimpleTimer, MemoryProfiler

with SimpleTimer("Complete experiment"):
    # Data loading
    with SimpleTimer("Data loading"):
        X_list = load_data()
    
    # Model training
    with MemoryProfiler("SGFA training"):
        results = run_sgfa(X_list)
    
    # Visualization
    with SimpleTimer("Visualization"):
        generate_plots(results)
```

### Command-line Profiling

Profile entire scripts from command line:

```bash
# Time profiling
python -m cProfile -o profile.stats run_experiments.py

# View results
python -m pstats profile.stats
>>> sort cumulative
>>> stats 20

# Or use optimization.profile_utils
python -c "
from optimization.profile_utils import profile_function
from experiments import run_data_validation

@profile_function(output_file='validation_profile.txt')
def main():
    run_data_validation()

main()
"
```

### Memory Profiling

For detailed memory profiling, use memory_profiler:

```bash
pip install memory-profiler

# Line-by-line memory profiling
python -m memory_profiler your_script.py

# Or use mprof
mprof run python run_experiments.py
mprof plot
```

## Best Practices

1. **Profile in production mode**: Use real MCMC parameters, not debug settings
2. **Profile bottlenecks**: Focus on the slowest parts identified by initial profiling
3. **Compare before/after**: Always profile before and after optimizations
4. **Use appropriate tools**:
   - `time_function`: Quick timing for development
   - `profile_function`: Detailed CPU profiling for optimization
   - `MemoryProfiler`: Memory usage tracking
   - `SimpleTimer`: Code block timing

## Example: Profiling an Experiment

```python
from optimization.profile_utils import time_function, SimpleTimer, MemoryProfiler
from experiments import SGFAParameterComparison
from data import load_qmap_pd

@time_function
def run_profiled_experiment():
    """Run SGFA parameter comparison with profiling."""
    
    # Load data with timing
    with SimpleTimer("Data loading"):
        data = load_qmap_pd()
        X_list = data['X_list']
    
    # Run experiment with memory tracking
    with MemoryProfiler("SGFA parameter comparison", verbose=True):
        with SimpleTimer("Parameter sweep"):
            config = ExperimentConfig(output_dir='./results')
            experiment = SGFAParameterComparison(config)
            results = experiment.run_parameter_sweep(X_list)
    
    return results

if __name__ == "__main__":
    results = run_profiled_experiment()
    print("Experiment complete!")
```

## Interpreting Results

### Timing Output

```
INFO:root:Data loading took 2.345s
INFO:root:SGFA parameter comparison in 125.678s
INFO:root:run_profiled_experiment took 128.123s
```

### Memory Output

```
INFO:root:Starting: SGFA training (Memory: 1234.5 MB)
INFO:root:Completed: SGFA training (Memory: 2345.6 MB, Change: +1111.1 MB)
```

### Profile Stats

```
         ncalls  tottime  percall  cumtime  percall filename:lineno(function)
          1      0.001    0.001   125.678  125.678 sgfa_hyperparameter_tuning.py:100(run_parameter_sweep)
         12      0.023    0.002   115.234    9.603 sparse_gfa.py:50(fit)
       1200      98.456   0.082    98.456    0.082 {built-in method jax...}
```

## See Also

- [optimization/profiler.py](profiler.py): Advanced profiling framework
- [optimization/memory_optimizer.py](memory_optimizer.py): Memory optimization
- [optimization/mcmc_optimizer.py](mcmc_optimizer.py): MCMC-specific optimization
