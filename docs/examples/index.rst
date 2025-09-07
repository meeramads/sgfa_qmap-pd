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

   from data.qmap_pd import load_qmap_pd_data
   from models.factory import create_model
   from analysis.model_runner import ModelRunner

   # Load qMAP-PD data
   X_list = load_qmap_pd_data()
   
   # Create sparse GFA model
   model = create_model('sparse_gfa', config={'K': 10, 'sparsity_level': 0.1})
   
   # Run MCMC inference
   runner = ModelRunner(model)
   results = runner.run_mcmc(X_list, num_samples=1000)

Model Comparison
----------------

.. code-block:: python

   from experiments.method_comparison import MethodComparisonExperiments
   
   # Compare different GFA variants
   exp = MethodComparisonExperiments(config)
   results = exp.run_gfa_variant_comparison(X_list)

Cross-Validation
----------------

.. code-block:: python

   from analysis.cross_validation import CVRunner
   
   # K-fold cross-validation
   cv = CVRunner(model, cv_type='kfold', k=5)
   cv_results = cv.run_cv(X_list)