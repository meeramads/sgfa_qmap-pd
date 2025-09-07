# Sparse Group Factor Analysis (SGFA) for qMAP-PD

Python implementation of Sparse Group Factor Analysis (SGFA) designed to identify latent disease factors that capture associations between various neuroimaging data modalities within Parkinson's disease population subgroups. This project applies and adapts this model to the qMAP-PD (https://qmaplab.com/qmap-pd) study dataset.

## Features

- **Multi-view Factor Analysis**: Handles multiple neuroimaging modalities simultaneously
- **Sparsity & Grouping**: Implements both sparse and group penalties for interpretable results
- **Performance Optimization**: Memory-efficient processing with automatic system optimization
- **Comprehensive Testing**: Extensive experimental validation framework
- **Clinical Validation**: Built-in tools for clinical subtype analysis and biomarker discovery

## Project Structure

### Core Modules
- **[get_data.py](get_data.py)**: High-level interface to load datasets or generate synthetic data
- **[run_analysis.py](run_analysis.py)**: Main script containing the SGFA model and experiment runner
- **[utils.py](utils.py)**: Utility functions supporting the analysis pipeline

### Data Management
- **[data/](data/)**: Data loading and generation modules
  - `qmap_pd.py`: qMAP-PD dataset loader with preprocessing
  - `synthetic.py`: Synthetic multi-view data generator for testing

### Analysis Pipeline  
- **[analysis/](analysis/)**: Core analysis components
  - `data_manager.py`: Data loading and preprocessing management
  - `config_manager.py`: Configuration and hyperparameter management
  - `model_runner.py`: Model training orchestration
  - `cross_validation.py`: Cross-validation framework

### Optimization & Performance
- **[performance/](performance/)**: Performance optimization framework
  - `memory_optimizer.py`: Memory management and monitoring
  - `data_streaming.py`: Efficient data loading for large datasets
  - `mcmc_optimizer.py`: Memory-optimized MCMC sampling
  - `profiler.py`: Performance profiling and benchmarking tools
  - `config.py`: Performance configuration management

### Experimental Validation
- **[experiments/](experiments/)**: Comprehensive experimental framework
  - `data_validation.py`: Data quality and preprocessing validation
  - `method_comparison.py`: SGFA vs traditional method comparison
  - `sensitivity_analysis.py`: Hyperparameter sensitivity testing
  - `reproducibility.py`: Reproducibility and robustness validation
  - `performance_benchmarks.py`: Scalability and efficiency benchmarking
  - `clinical_validation.py`: Clinical subtype and biomarker validation

### Visualization & Models
- **[visualization/](visualization/)**: Result visualization and diagnostic plots
- **[models/](models/)**: Model definitions and variants
- **[tests/](tests/)**: Comprehensive test suite

## Installation

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/meeramads/sgfa_qmap-pd
cd sgfa_qmap-pd

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Development Installation
```bash
# Install with development tools
pip install -r requirements-dev.txt

# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

## Quick Start

### Basic Analysis
```python
from get_data import qmap_pd, generate_synthetic_data
from run_analysis import run_sgfa_analysis

# Load data
X_list, metadata = qmap_pd()  # Or generate_synthetic_data()

# Run SGFA analysis
results = run_sgfa_analysis(X_list, K=5, sparsity_level=0.3)
```

### With Performance Optimization
```python
from performance import auto_configure_for_system, PerformanceManager

# Auto-configure for your system
config = auto_configure_for_system()

# Run with optimization
with PerformanceManager(config) as manager:
    results = run_sgfa_analysis(X_list, K=5, sparsity_level=0.3)
    print(f"Memory used: {manager.get_memory_status()}")
```

### Comprehensive Experiments
```python
from experiments import ExperimentConfig, DataValidationExperiments

# Configure experiments
config = ExperimentConfig(output_dir="./results", save_plots=True)

# Run validation experiments
validator = DataValidationExperiments(config)
results = validator.run_comprehensive_data_validation(X_list)
```

## Performance Optimization

The project includes a comprehensive performance optimization framework:

### Memory Management
- **Adaptive memory limits** based on available system resources
- **Real-time monitoring** with automatic cleanup
- **Memory-efficient data structures** and array optimization

### Data Processing
- **Chunked processing** for large datasets (>1GB)
- **HDF5 support** with compression
- **Streaming data loaders** to minimize memory footprint

### MCMC Optimization
- **Memory-efficient sampling** strategies
- **Adaptive batching** for large datasets
- **Gradient checkpointing** to reduce memory usage

### Quick Performance Setup
```python
from performance.config import auto_configure_for_system

# Automatically optimize for your system
config = auto_configure_for_system()

# Or use predefined presets
config = PerformanceConfig().create_preset('memory_efficient')  # For limited RAM
config = PerformanceConfig().create_preset('fast')              # For speed
config = PerformanceConfig().create_preset('balanced')          # Balanced approach
```

## Experimental Validation

The framework includes comprehensive validation experiments:

### Data Validation
- Quality assessment and outlier detection
- Preprocessing strategy comparison
- Multi-view data alignment validation

### Method Comparison  
- SGFA variants (sparse vs group vs standard)
- Traditional methods (PCA, ICA, Factor Analysis)
- Scalability and performance benchmarking

### Clinical Validation
- PD subtype classification validation
- Disease progression prediction
- Biomarker discovery and validation
- External cohort generalization testing

### Run Experiments
```bash
# Run comprehensive validation
python -m experiments.run_all_experiments

# Run specific experiment
python -c "from experiments import DataValidationExperiments; ..."
```

## Running on Colab (GPU)

For GPU-accelerated training:

1. Open `run-Colab_GPU.ipynb` in Google Colab
2. Go to Runtime → Change runtime type → Select GPU hardware accelerator
3. Run the setup cells to configure the environment
4. Train the model:
```bash
!py310cuda run_analysis.py --device gpu --K 5 --sparsity_level 0.3
```

## Testing

Run the comprehensive test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test modules
pytest tests/test_data/
pytest tests/test_analysis/
pytest tests/test_performance/
```

## Configuration

### Basic Configuration
```python
from analysis.config_manager import ConfigManager

config = ConfigManager()
config.set_model_params(K=5, sparsity_level=0.3)
config.set_mcmc_params(num_samples=1000, num_chains=4)
```

### Performance Configuration
```yaml
# config.yaml
memory:
  max_memory_gb: 8.0
  warning_threshold: 0.8
  enable_monitoring: true

data:
  enable_chunking: true
  memory_limit_gb: 4.0
  enable_compression: true

mcmc:
  enable_adaptive_batching: true
  enable_checkpointing: true
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Run linting (`black . && flake8`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- qMAP-PD Study: https://qmaplab.com/qmap-pd
- JAX Documentation: https://jax.readthedocs.io/
- NumPyro Documentation: https://num.pyro.ai/