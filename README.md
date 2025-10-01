# Sparse Group Factor Analysis (SGFA) for qMAP-PD

Python implementation of Sparse Group Factor Analysis (SGFA) designed to identify latent disease factors that capture associations between various neuroimaging data modalities within Parkinson's disease population subgroups. This project applies and adapts this model to the qMAP-PD (<https://qmaplab.com/qmap-pd>) study dataset.

## Features

- **Multi-view Factor Analysis**: Handles multiple neuroimaging modalities simultaneously
- **Sparsity & Grouping**: Implements both sparse and group penalties for interpretable results
- **Computational Optimization**: Memory-efficient processing with automatic system optimization and performance mixins
- **Comprehensive Testing**: Extensive experimental validation framework
- **Clinical Validation**: Built-in tools for clinical subtype analysis and biomarker discovery

## Project Structure

### Core Modules

- **[core/get_data.py](core/get_data.py)**: High-level interface to load datasets or generate synthetic data
- **[core/run_analysis.py](core/run_analysis.py)**: Main script containing the SGFA model and experiment runner
- **[core/utils.py](core/utils.py)**: Utility functions supporting the analysis pipeline

### Data Management

- **[data/](data/)**: Data loading and generation modules
  - `qmap_pd.py`: qMAP-PD dataset loader with preprocessing
  - `synthetic.py`: Synthetic multi-view data generator for testing

### Analysis Pipeline

- **[analysis/](analysis/)**: Core analysis components
  - `data_manager.py`: Data loading and preprocessing management **IMPLEMENTED**
  - `config_manager.py`: Configuration and hyperparameter management **IMPLEMENTED**
  - `model_runner.py`: Model training orchestration **IMPLEMENTED**
  - `cross_validation.py`: Cross-validation framework **IMPLEMENTED**
  - `cross_validation_library.py`: Advanced neuroimaging CV **FRAMEWORK ONLY - NEEDS DEVELOPMENT**
  - `cv_fallbacks.py`: Reusable fallback utilities for advanced CV features **IMPLEMENTED**
  - `experiment_utils.py`: Bridge utilities for experiments to use analysis pipeline **IMPLEMENTED**

### Computational Optimization

- **[optimization/](optimization/)**: Computational optimization framework (renamed from `performance/` to avoid confusion with model performance)
  - `memory_optimizer.py`: Memory management and monitoring with automatic cleanup
  - `data_streaming.py`: Memory-efficient data loading for large datasets
  - `mcmc_optimizer.py`: Memory-optimized MCMC sampling with checkpointing
  - `profiler.py`: Performance profiling and benchmarking tools
  - `config.py`: Auto-configuration for different system capabilities
  - `experiment_mixins.py`: Reusable optimization patterns for experiments **NEW**

### Experimental Validation

- **[experiments/](experiments/)**: Comprehensive experimental framework with advanced neuroimaging CV
  - `data_validation.py`: Data quality and preprocessing validation **IMPLEMENTED AND TESTED**
  - `model_comparison.py`: SGFA vs traditional method comparison **IMPLEMENTED AND TESTED** + **NeuroImagingMetrics NEEDS DEVELOPMENT**
  - `sgfa_parameter_comparison.py`: SGFA variant comparison **IMPLEMENTED AND TESTED** + **HyperOptimizer NEEDS DEVELOPMENT**
  - `sensitivity_analysis.py`: Hyperparameter sensitivity testing **IMPLEMENTED AND TESTED**
  - `reproducibility.py`: Reproducibility and robustness validation **IMPLEMENTED AND TESTED**
  - `clinical_validation.py`: Clinical subtype validation + basic SGFA benchmarks **IMPLEMENTED AND TESTED** + **Advanced neuroimaging CV NEEDS DEVELOPMENT**
  - `debug_experiments.py`: Lightweight testing framework for rapid validation **IMPLEMENTED**

### Models & Implementation

- **[models/](models/)**: Core model implementations
  - `base.py`: Base model class and shared functionality **IMPLEMENTED**
  - `standard_gfa.py`: Standard Group Factor Analysis with ARD priors **HIGH GPU USAGE**
  - `sparse_gfa.py`: Sparse GFA with regularized horseshoe priors **IMPLEMENTED**
  - `latent_class_analysis.py`: Patient subtyping and clustering models **VERY HIGH GPU USAGE**
  - `factory.py`: Model factory for instantiation and configuration **IMPLEMENTED**
  - `variants/neuroimaging_gfa.py`: Specialized neuroimaging variants **NEEDS DEVELOPMENT**

### Visualization & Testing

- **[visualization/](visualization/)**: Result visualization and diagnostic plots **IMPLEMENTED AND TESTED**
  - `factor_plots.py`: Factor analysis visualizations with brain mapping
  - `brain_plots.py`: Neuroimaging-specific brain factor maps
  - `report_generator.py`: Comprehensive HTML analysis reports
- **[tests/](tests/)**: Comprehensive test suite

## Automatic Optimization Features

All experiments now include **automatic computational optimization** through the `@performance_optimized_experiment` decorator:

### **Memory Optimization**

- **Automatic array optimization**: Converts float64→float32 when precision allows
- **Real-time memory monitoring**: Tracks usage with automatic cleanup
- **Adaptive batch sizing**: Calculates optimal batch sizes based on available memory
- **Memory pressure handling**: Automatic intervention when memory limits approached

### **Data Streaming**

- **Large dataset support**: Memory-efficient streaming for datasets larger than RAM
- **Chunked processing**: Automatic data chunking for cross-validation and analysis
- **Memory-efficient contexts**: Automatic memory management during processing

### **MCMC Optimization**

- **Gradient checkpointing**: Reduces memory usage during MCMC sampling
- **Adaptive subsampling**: Automatic data subsampling when memory constrained
- **Batch sampling**: Memory-optimized sampling for large datasets

### **System Auto-Configuration**

- **Hardware detection**: Automatically detects system capabilities (RAM, CPU)
- **Optimization strategy selection**: Chooses optimal settings based on available resources
- **Performance monitoring**: Real-time tracking of computational performance

### **Usage**

```python
# All experiments automatically optimized - no code changes needed!
from experiments import ClinicalValidationExperiments

config = ExperimentConfig(auto_configure_system=True)  # Enable auto-optimization
experiment = ClinicalValidationExperiments(config)
# Memory optimization, streaming, and monitoring happen automatically
```

## Development Status & Warnings

### **Features Requiring Development**

The following features have **framework infrastructure** in place but require **substantial development work**:

#### **Advanced Cross-Validation (High Priority)**

- **ClinicalAwareSplitter**: Clinical-aware CV splitting logic
- **NeuroImagingMetrics**: Domain-specific evaluation metrics
- **NeuroImagingHyperOptimizer**: Bayesian hyperparameter optimization
- **Status**: Framework ready, ~600-1000 lines of specialized code needed
- **Estimated Effort**: 2-4 weeks for experienced ML developer

#### **NeuroGFA Models (Research Priority)**

- **neuroimaging_gfa.py**: Specialized spatial priors for neuroimaging
- **Status**: Framework ready, requires deep neuroimaging + Bayesian expertise
- **Estimated Effort**: 1-3 months for neuroimaging researcher

### **GPU Memory Warnings**

**CAUTION**: The following models have **high computational requirements**:

#### **Standard GFA** (`standard_gfa.py`)

- **GPU Memory**: High usage (8-16GB+ depending on data size)
- **Recommendation**: Use sparse_gfa for most applications
- **Alternative**: Reduce K (number of factors) or data size

#### **Latent Class Analysis** (`latent_class_analysis.py`)

- **GPU Memory**: Very high usage (16GB+ typical, may exceed most GPU limits)
- **Recommendation**: Use for small datasets only or powerful GPU clusters
- **Alternative**: Use clustering post-hoc on sparse_gfa factors

#### **Safe Default**: `sparse_gfa.py`

- **GPU Memory**: Moderate usage (2-8GB typical)
- **Recommendation**: Start here for most neuroimaging applications
- **Performance**: Excellent for PD subtype analysis

### **Planning Frameworks**

#### Advanced CV Development Plan

#### Phase 1: ClinicalAwareSplitter (1-2 weeks)

```python
# Core functionality needed:
class ClinicalAwareSplitter:
    def split(self, X, y, groups, clinical_data):
        # 1. Multi-variable stratification logic
        # 2. Site/scanner awareness
        # 3. Group preservation (same subject not in train/test)
        # 4. Minimum sample size constraints
        # 5. Demographics balance validation
```

#### Phase 2: NeuroImagingMetrics (1-2 weeks)

```python
# Core functionality needed:
class NeuroImagingMetrics:
    def calculate_fold_metrics(self, train_result, test_metrics, clinical_data):
        # 1. Factor interpretability scoring
        # 2. Cross-view consistency metrics
        # 3. Clinical association scoring
        # 4. Spatial coherence measures
        # 5. Composite scoring logic

    def evaluate_model_comparison(self, X_data, model_outputs, model_info):
        # 1. Multi-model evaluation framework
        # 2. Statistical significance testing
        # 3. Effect size calculations
```

#### Phase 3: NeuroImagingHyperOptimizer (2-3 weeks)

```python
# Core functionality needed:
class NeuroImagingHyperOptimizer:
    def optimize_hyperparameters(self, X_data, clinical_data, search_space):
        # 1. Bayesian optimization setup (optuna/hyperopt)
        # 2. Clinical-aware objective functions
        # 3. Multi-objective optimization (accuracy + interpretability)
        # 4. Parameter importance analysis
        # 5. Convergence tracking and early stopping
```

**Dependencies**: sklearn, optuna/hyperopt, scipy.stats
**Testing Requirements**: Synthetic + real clinical data validation

---

#### NeuroGFA Development Plan

#### Phase 1: Spatial Prior Framework (3-4 weeks)

```python
# Core research needed:
class NeuroImagingGFA:
    def __init__(self, spatial_prior_type="ising", anatomical_connectivity=None):
        # 1. Ising model priors for spatial coherence
        # 2. Anatomical connectivity integration
        # 3. Multi-scale spatial modeling
        # 4. ROI-aware factor allocation
```

#### Phase 2: Advanced Neuroimaging Features (4-6 weeks)

```python
# Advanced functionality:
- Atlas-based factor initialization
- Symmetric brain modeling
- Multi-modal data fusion (structural + functional)
- Temporal factor evolution modeling
- Group-level + subject-specific factors
```

#### Phase 3: Clinical Integration (2-3 weeks)

```python
# Clinical neuroimaging integration:
- Disease progression modeling
- Biomarker discovery optimization
- Multi-site harmonization
- Scanner-invariant factors
```

**Dependencies**: nibabel, nilearn, advanced Bayesian modeling, neuroimaging expertise
**Data Requirements**: Multi-modal neuroimaging datasets with anatomical atlases

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
pip install -e .
```

### Development Installation

```bash
# Install with development tools
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

## Quick Start

### Debug Testing Framework

For rapid validation and development testing, use the debug experiment runner:

```bash
# Individual debug experiments (reduced MCMC parameters for speed)
python debug_experiments.py data_validation
python debug_experiments.py sgfa_parameter_comparison
python debug_experiments.py model_comparison
python debug_experiments.py sensitivity_analysis
python debug_experiments.py clinical_validation
python debug_experiments.py reproducibility

# Test all components
python debug_experiments.py all
```

**Debug vs Production Parameters:**

- **Debug**: 50-150 MCMC samples, 3-5 factors, minimal preprocessing
- **Production**: 1000+ MCMC samples, full factors, comprehensive preprocessing
- **Runtime**: Varies significantly based on system specifications and data complexity

### Basic Analysis

```python
from core.get_data import qmap_pd, generate_synthetic_data
from core.run_analysis import run_sgfa_analysis

# Load data - Option 1: Real qMAP-PD data
# X_list, metadata = qmap_pd(data_dir='path/to/data')

# Load data - Option 2: Synthetic data
data = generate_synthetic_data(num_sources=3, K=5)
X_list = data['X_list']

# Run SGFA analysis
results = run_sgfa_analysis(X_list, K=5, sparsity_level=0.3)
```

### Using Analysis Pipeline Components

```python
from analysis import quick_sgfa_run, create_analysis_components

# Quick SGFA run with minimal setup
results = quick_sgfa_run(X_list, K=5, percW=25.0)

# Or use the full pipeline components
data_manager, model_runner = create_analysis_components({
    'K': 5, 'num_sources': 3, 'dataset': 'synthetic'
})
X_list, hypers = data_manager.load_data()
results = model_runner.run_standard_analysis(X_list, hypers, {'X_list': X_list})
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
from experiments import (
    ExperimentConfig,
    DataValidationExperiments,
    ModelArchitectureComparison,
    SGFAParameterComparison,
    SensitivityAnalysisExperiments,
    ReproducibilityExperiments,
    PerformanceBenchmarkExperiments,
    ClinicalValidationExperiments
)

# Configure experiments
config = ExperimentConfig(output_dir="./results", save_plots=True)

# Run data validation experiments
data_validator = DataValidationExperiments(config)
data_results = data_validator.run_comprehensive_data_validation(X_list)

# Run model comparison experiments
model_comparison = ModelArchitectureComparison(config)
comparison_results = model_comparison.run_full_comparison(X_list)

# Run SGFA parameter optimization
sgfa_optimizer = SGFAParameterComparison(config)
param_results = sgfa_optimizer.run_parameter_sweep(X_list)

# Run sensitivity analysis
sensitivity = SensitivityAnalysisExperiments(config)
sensitivity_results = sensitivity.run_sensitivity_analysis(X_list)

# Run reproducibility validation
reproducibility = ReproducibilityExperiments(config)
repro_results = reproducibility.run_reproducibility_tests(X_list)

# Run clinical validation with integrated SGFA performance benchmarks
clinical = ClinicalValidationExperiments(config)
perf_results = clinical.run_sgfa_clinical_validation(X_list)

# Basic clinical validation (subtype classification)
clinical_results = clinical.run_clinical_validation(X_list, clinical_data)

# Advanced: Clinical-stratified cross-validation
# Configure validation types to include clinical-stratified CV
config_dict = {
    "clinical_validation": {
        "validation_types": ["subtype_classification", "clinical_stratified_cv"],
        "classification_metrics": ["accuracy", "precision", "recall", "f1_score"]
    }
}
clinical_results_advanced = clinical.run_clinical_validation(X_list, clinical_data, config_dict)
```

### Advanced Neuroimaging Cross-Validation

**NEW**: Clinical-stratified CV now available! Basic neuroimaging CV features work, advanced features require development:

```python
# THESE WORK - Basic SGFA + clinical validation is implemented:
from experiments.clinical_validation import ClinicalValidationExperiments

clinical_validator = ClinicalValidationExperiments(config)

# Integrated SGFA + clinical validation benchmarks
validation_results = clinical_validator.run_sgfa_clinical_validation(
    X_base=X_list, hypers={"percW": 25.0}, args={"K": 5, "model": "sparseGFA"}
)

# For pure SGFA scalability analysis, use sgfa_parameter_comparison:
from experiments.sgfa_parameter_comparison import SGFAParameterComparison

sgfa_analyzer = SGFAParameterComparison(config)
scalability_results = sgfa_analyzer.run_comprehensive_sgfa_scalability_analysis(
    X_list=X_list, hypers={"percW": 25.0}, args={"K": 5}
)

# Memory/performance metrics included in comprehensive analysis above

# Basic clinical validation (implemented, advanced features need development):
cv_results = clinical_validator.run_sgfa_clinical_validation(
    X_base=X_list,
    hypers={"percW": 25.0},
    args={"K": 5, "model": "sparseGFA"},
    clinical_data=clinical_data
)

# Neuroimaging hyperparameter optimization
from experiments.sgfa_parameter_comparison import SGFAParameterComparison

sgfa_exp = SGFAParameterComparison(config)
# This will fail - NeuroImagingHyperOptimizer not implemented
hyperopt_results = sgfa_exp.run_neuroimaging_hyperparameter_optimization(
    X_list=X_list,
    hypers={"percW": 25.0},
    args={"K": 5},
    clinical_data=clinical_data
)

# NEW: Clinical-stratified cross-validation works!
from experiments.clinical_validation import ClinicalValidationExperiments

clinical_exp = ClinicalValidationExperiments(config)

# Configure for clinical-stratified CV
config_dict = {
    "clinical_validation": {
        "validation_types": ["clinical_stratified_cv"],
        "cross_validation": {"n_folds": 5, "stratified": True}
    }
}

# This works - Clinical-stratified CV with robust neuroimaging scaling
clinical_stratified_results = clinical_exp.run_clinical_validation(
    X_list=X_list,
    clinical_data=clinical_data,
    config=config_dict
)
```

#### Advanced CV Features Status

- **ClinicalAwareSplitter**: **Basic implementation available** - Clinical stratification with robust neuroimaging scaling
- **NeuroImagingMetrics**: Infrastructure exists, needs specialized neuroimaging evaluation metrics
- **NeuroImagingHyperOptimizer**: Infrastructure exists, needs Bayesian optimization logic
- **Clinical-Stratified Validation**: **Available** - Use `validation_types: ["clinical_stratified_cv"]`

#### Working Alternative: Basic Cross-Validation

The experiments automatically fall back to working basic CV when advanced features fail:

```python
from analysis.cross_validation import CVRunner

# This works - CVRunner automatically falls back to basic CV
cv_runner = CVRunner(config, results_dir="./results")
cv_results, cv_obj = cv_runner.run_cv_analysis(
    X_list=X_list,
    hypers={"percW": 25.0},
    data={"K": 5, "model": "sparseGFA"}
)

# Basic CV uses SparseBayesianGFACrossValidator with sklearn-style splitting
# but adapted for SGFA models
```

## Performance Optimization

The project includes a comprehensive performance optimization framework:

### Memory Management

- **Adaptive memory limits** based on available system resources
- **Real-time monitoring** with automatic cleanup and GPU memory management
- **Memory-efficient data structures** and array optimization
- **Enhanced GPU cleanup** with JAX cache clearing and aggressive garbage collection

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
from performance.config import PerformanceConfig
config = PerformanceConfig().create_preset('memory_efficient')  # For limited RAM
config = PerformanceConfig().create_preset('fast')              # For speed
config = PerformanceConfig().create_preset('balanced')          # Balanced approach
```

## Validation Experiments

The framework includes comprehensive validation experiments:

### Data Validation

- Quality assessment and outlier detection
- Preprocessing strategy comparison
- Multi-view data alignment validation

### Method Comparison

- SGFA variants (sparse vs group vs standard)
- Traditional methods (PCA, ICA, Factor Analysis)
- SGFA parameter comparison and optimization
- Scalability and performance benchmarking

### Clinical Validation

- **PD subtype classification validation** **IMPLEMENTED**
- **Clinical-stratified cross-validation** **AVAILABLE** - CV with proper clinical stratification and robust neuroimaging scaling
- Disease progression prediction **PLACEHOLDER**
- Biomarker discovery and validation **PLACEHOLDER**
- External cohort generalization testing **FUTURE WORK**

#### Clinical-Stratified CV Features

**Currently Available:**

- CV folds stratified by clinical variables (diagnosis, subtypes, etc.)
- Robust scaling (median/MAD) optimized for neuroimaging data
- Enhanced convergence checking and timeout handling
- Professional error handling with automatic fallbacks

**Future Development (Advanced Neuroimaging CV):**

- Neuroimaging-specific priors and spatial structure modeling
- Scanner/site effect correction and harmonization
- PD-specific disease progression constraints
- Brain connectivity and anatomical structure integration

### Run Experiments

#### Production Experiments

```bash
# Run all production experiments (full MCMC parameters)
python run_experiments.py --config config.yaml --experiments all

# Run specific production experiments
python run_experiments.py --config config.yaml --experiments data_validation model_comparison

# Run with custom data directory
python run_experiments.py --config config.yaml --data-dir /path/to/data
```

#### Debug Experiments

```bash
# Run all debug experiments (reduced parameters for testing)
python debug_experiments.py all

# Run specific debug experiments
python debug_experiments.py sgfa_parameter_comparison
python debug_experiments.py model_comparison
```

**Available Experiment Types:**

**FULLY IMPLEMENTED AND TESTED:**

- `data_validation`: Data quality and preprocessing validation
- `sgfa_parameter_comparison`: SGFA parameter optimization and scalability analysis
- `model_comparison`: SGFA vs traditional methods (PCA, ICA, FA)
- `sensitivity_analysis`: Hyperparameter sensitivity testing
- `reproducibility`: Reproducibility and robustness validation
- `clinical_validation`: Clinical subtype validation + basic SGFA benchmarks

**ADVANCED FEATURES NEED DEVELOPMENT:**

- Advanced neuroimaging-specific hyperparameter optimization
- Advanced clinical-aware cross-validation benchmarks
- Sophisticated neuroimaging metrics and validation
- Advanced neuroimaging CV methods (ClinicalAwareSplitter, NeuroImagingMetrics)

#### Run specific experiments directly (programmatic)

```python
from experiments import DataValidationExperiments, ExperimentConfig
config = ExperimentConfig(output_dir='./results')
validator = DataValidationExperiments(config)
# Add your data and run experiments
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test modules
pytest tests/data/
pytest tests/analysis/
pytest tests/models/
pytest tests/experiments/
pytest tests/visualization/
```

## Configuration

### Configuration Structure

The project uses YAML configuration files with validated schemas:

```yaml
# config.yaml - Production configuration
data:
  data_dir: "./qMAP-PD_data"
  clinical_file: "data_clinical/clinical.tsv"
  volume_dir: "volume_matrices"
  imaging_as_single_view: true

experiments:
  base_output_dir: "./results"
  save_intermediate: true
  generate_plots: true
  max_parallel_jobs: 2

model:
  model_type: "sparse_gfa"
  K: 5
  sparsity_lambda: 0.3
  num_samples: 1000
  num_chains: 1
```

```yaml
# config_debug.yaml - Debug configuration (reduced parameters)
experiments:
  base_output_dir: "./debug_results"
  save_intermediate: false
  generate_plots: false

model:
  K: 3
  num_samples: 50
  # ... other reduced parameters
```

**Note:** The `run_experiments` field was removed from the configuration schema for validation compatibility.

### Basic Configuration

```python
from core.config_utils import get_default_configuration, validate_configuration
import yaml

# Get default configuration
config = get_default_configuration()

# Modify as needed
config['model']['K'] = 5
config['model']['sparsity_lambda'] = 0.3
config['model']['num_samples'] = 1000
config['model']['num_chains'] = 4

# Validate configuration
if validate_configuration(config):
    print("Configuration is valid!")
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

This project is licensed under the MIT License.

## References

- qMAP-PD Study: <https://qmaplab.com/qmap-pd>
- JAX Documentation: <https://jax.readthedocs.io/>
- NumPyro Documentation: <https://num.pyro.ai/>
