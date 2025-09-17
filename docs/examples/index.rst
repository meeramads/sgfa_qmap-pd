Examples
========

.. toctree::
   :maxdepth: 2

   basic_usage
   clinical_analysis
   performance_optimization

Basic Usage Examples
====================

Quick Start
-----------

.. code-block:: python

   from core.get_data import generate_synthetic_data
   from analysis import quick_sgfa_run

   # Generate synthetic data
   data = generate_synthetic_data(num_sources=3, K=5)
   X_list = data['X_list']

   # Quick SGFA analysis
   results = quick_sgfa_run(X_list, K=5, percW=25.0)

Using Analysis Pipeline Components
----------------------------------

.. code-block:: python

   from analysis import create_analysis_components

   # Create pipeline components
   data_manager, model_runner = create_analysis_components({
       'K': 5, 'num_sources': 3, 'dataset': 'synthetic'
   })

   # Load and prepare data
   data = data_manager.load_data()
   X_list, hypers = data_manager.prepare_for_analysis(data)

   # Run analysis
   results = model_runner.run_standard_analysis(X_list, hypers, data)

Model Comparison
----------------

.. code-block:: python

   from experiments.model_comparison import ModelArchitectureComparison
   from experiments.framework import ExperimentConfig

   # Compare different GFA variants
   config = ExperimentConfig(output_dir="./results")
   exp = ModelArchitectureComparison(config)
   results = exp.run_sgfa_variant_comparison(X_list, hypers={}, args={})

Cross-Validation
----------------

.. code-block:: python

   from analysis.cross_validation import CVRunner

   # Cross-validation with fallback support
   cv_runner = CVRunner(config, results_dir="./results")
   cv_results, cv_obj = cv_runner.run_cv_analysis(
       X_list=X_list,
       hypers={"percW": 25.0},
       data={"K": 5, "model": "sparseGFA"}
   )