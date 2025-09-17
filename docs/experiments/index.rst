Experimental Framework
======================

.. toctree::
   :maxdepth: 2

   data_validation
   method_comparison
   clinical_validation

The experimental framework provides comprehensive validation capabilities for SGFA analysis on qMAP-PD data.

Framework Overview
==================

The experimental framework is built around the ``ExperimentFramework`` base class and provides:

* **Data Validation**: Quality assessment and preprocessing validation
* **Method Comparison**: Benchmarking different GFA variants
* **Sensitivity Analysis**: Parameter robustness testing  
* **Reproducibility Testing**: Ensuring consistent results
* **Performance Benchmarks**: Scalability and efficiency testing
* **Clinical Validation**: PD subtype classification validation

Data Validation Experiments
============================

.. code-block:: python

   from experiments.data_validation import DataValidationExperiments
   from experiments.framework import ExperimentConfig

   config = ExperimentConfig(output_dir="./results")
   exp = DataValidationExperiments(config)
   results = exp.run_comprehensive_data_validation(X_list)

Method Comparison Experiments
=============================

.. code-block:: python

   from experiments.model_comparison import ModelArchitectureComparison
   from experiments.framework import ExperimentConfig

   config = ExperimentConfig(output_dir="./results")
   exp = ModelArchitectureComparison(config)
   results = exp.run_full_comparison(X_list)

Clinical Validation Experiments
===============================

.. code-block:: python

   from experiments.clinical_validation import ClinicalValidationExperiments
   from experiments.framework import ExperimentConfig

   config = ExperimentConfig(output_dir="./results")
   exp = ClinicalValidationExperiments(config)
   # Uses fallback CV when advanced features unavailable
   results = exp.run_clinical_validation(X_list, clinical_data)

Running All Experiments
========================

.. code-block:: python

   # Using the experiment runner
   python run_experiments.py --config config.yaml --experiments all

   # Or run specific experiments
   python run_experiments.py --config config.yaml --experiments data_validation model_comparison