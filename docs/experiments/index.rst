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
   
   exp = DataValidationExperiments(config)
   results = exp.run_data_quality_assessment(X_list)

Method Comparison Experiments
=============================

.. code-block:: python

   from experiments.method_comparison import MethodComparisonExperiments
   
   exp = MethodComparisonExperiments(config)
   results = exp.run_gfa_variant_comparison(X_list)

Clinical Validation Experiments
===============================

.. code-block:: python

   from experiments.clinical_validation import ClinicalValidationExperiments
   
   exp = ClinicalValidationExperiments(config)
   results = exp.run_pd_subtype_classification(X_list, clinical_labels)