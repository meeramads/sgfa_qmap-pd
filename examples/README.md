# SGFA qMAP-PD Examples

This directory contains example scripts demonstrating how to use the SGFA qMAP-PD framework.

## Quick Start

**New to SGFA qMAP-PD?** Start here:

```bash
# Run the quick start guide
python examples/quick_start.py

# Run your first analysis (fast debug mode)
python debug_experiments.py data_validation
```

## Example Scripts

### 🚀 Getting Started

- **[quick_start.py](quick_start.py)** - Quickest way to get started
  - Synthetic data example
  - Command-line examples
  - Python API examples
  - Real data loading
  - Next steps guidance

### ⚙️ Configuration

- **[experiment_configuration.py](experiment_configuration.py)** - Advanced configuration options
  - ROI selection (`--select-rois`)
  - Clinical feature exclusion (`--exclude-clinical`)
  - K value selection (`--test-k`)
  - Combined options
  - Typical workflows

### 📊 Data Handling

- **[data_example.py](data_example.py)** - Data loading and preprocessing
  - Synthetic data generation
  - qMAP-PD data loading
  - Preprocessing strategies
  - Data validation
  - ROI selection
  - Clinical feature exclusion

### 🤖 Models

- **[models_example.py](models_example.py)** - Model configuration and usage
  - Model factory usage
  - Sparse GFA models
  - Standard GFA models
  - Model comparison
  - Hyperparameter configuration

### 📈 Visualization

- **[visualization_example.py](visualization_example.py)** - Result visualization
  - Factor plots
  - Brain maps
  - Diagnostic plots
  - Report generation

### ⚡ Performance

- **[performance_example.py](performance_example.py)** - Performance optimization
  - Memory management
  - MCMC optimization
  - Data streaming
  - Profiling

## Usage Patterns

### Pattern 1: Quick Analysis

```bash
# Generate synthetic data and run SGFA
python examples/quick_start.py
```

### Pattern 2: Configure and Run

```bash
# Configure your experiment
python examples/experiment_configuration.py  # Read examples

# Run with your configuration
python run_experiments.py \
    --experiments sgfa_parameter_comparison \
    --select-rois volume_sn_voxels.tsv \
    --exclude-clinical age sex tiv \
    --test-k 2 3
```

### Pattern 3: Explore Data

```bash
# Learn about data loading
python examples/data_example.py

# Run data validation
python run_experiments.py --experiments data_validation
```

### Pattern 4: Model Development

```bash
# Learn about models
python examples/models_example.py

# Compare models
python run_experiments.py --experiments model_comparison
```

### Pattern 5: Visualization

```bash
# Learn about visualization options
python examples/visualization_example.py

# Results are automatically visualized in experiments
# Check ./results/*/plots/ for generated visualizations
```

## Example Workflows

### Workflow 1: First-Time User

```bash
# 1. Quick start
python examples/quick_start.py

# 2. Try debug mode (fast)
python debug_experiments.py data_validation

# 3. Run full experiment
python run_experiments.py --experiments data_validation
```

### Workflow 2: PD Subtype Analysis

```bash
# 1. Understand configuration options
python examples/experiment_configuration.py

# 2. Validate data
python run_experiments.py --experiments data_validation \
    --select-rois volume_sn_voxels.tsv

# 3. Optimize parameters
python run_experiments.py --experiments sgfa_parameter_comparison \
    --select-rois volume_sn_voxels.tsv \
    --exclude-clinical age sex tiv \
    --test-k 2 3

# 4. Clinical validation
python run_experiments.py --experiments clinical_validation \
    --select-rois volume_sn_voxels.tsv \
    --exclude-clinical age sex tiv
```

### Workflow 3: Multi-ROI Analysis

```bash
# 1. Configure for 4 views (clinical + 3 ROIs)
python run_experiments.py --experiments sgfa_parameter_comparison \
    --select-rois volume_sn_voxels.tsv volume_putamen_voxels.tsv volume_lentiform_voxels.tsv \
    --test-k 3 4 5

# 2. Compare results across configurations
python run_experiments.py --experiments model_comparison \
    --select-rois volume_sn_voxels.tsv volume_putamen_voxels.tsv volume_lentiform_voxels.tsv
```

## Running Examples

### Prerequisites

```bash
# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Running Individual Examples

```bash
# Make examples executable
chmod +x examples/*.py

# Run any example
python examples/quick_start.py
python examples/data_example.py
python examples/experiment_configuration.py
```

### Running All Examples

```bash
# Run all examples sequentially
for example in examples/*.py; do
    if [ "$example" != "examples/__init__.py" ]; then
        echo "Running $example..."
        python "$example"
    fi
done
```

## Example Output

Each example produces formatted output explaining:
- What it's demonstrating
- Step-by-step process
- Expected results
- Tips and recommendations
- Next steps

Example output structure:
```
==================================================
EXAMPLE TITLE
==================================================

1️⃣ Step 1: Description
   [Output and results]

2️⃣ Step 2: Description
   [Output and results]

✅ Complete! Summary and next steps.
==================================================
```

## Tips for Learning

1. **Start Simple**: Begin with `quick_start.py`
2. **Read Comments**: Examples are heavily commented
3. **Experiment**: Modify parameters and rerun
4. **Debug Mode**: Use `debug_experiments.py` for fast testing
5. **Check Results**: Look in `./results/` for outputs

## Common Questions

### Q: Which example should I start with?

A: Start with `quick_start.py` for the basics, then `experiment_configuration.py` for your specific needs.

### Q: How do I use my own data?

A: See `data_example.py` for data loading examples, or use command-line flags:
```bash
python run_experiments.py --data-dir /path/to/your/data
```

### Q: Examples run slowly?

A: Use debug mode for faster testing:
```bash
python debug_experiments.py [experiment_name]
```

### Q: Where are results saved?

A: Results are saved to `./results/` with timestamped directories. Directory names include your configuration for easy identification.

### Q: Can I modify these examples?

A: Yes! Copy an example and modify it for your needs. They're designed to be starting points.

## Additional Resources

- **[README.md](../README.md)**: Complete project documentation
- **[docs/configuration.md](../docs/configuration.md)**: Detailed configuration guide
- **[docs/TESTING.md](../docs/TESTING.md)**: Testing framework
- **[config.yaml](../config.yaml)**: Main configuration file with all options

## Need Help?

- Run with `-h` or `--help` flag for command-line options
- Check docstrings: `python -m pydoc module_name`
- Run tests: `pytest`
- Report issues: GitHub Issues

## Contributing Examples

Have a useful example? Consider adding it:
1. Follow the existing format
2. Include clear comments and docstrings
3. Add to this README
4. Test it works standalone

---

**Ready to analyze PD subtypes!** 🧠🔬
