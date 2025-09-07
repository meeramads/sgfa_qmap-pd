SGFA qMAP-PD Documentation
==========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules
   examples/index
   experiments/index

Welcome to SGFA qMAP-PD
=======================

This package implements Sparse Group Factor Analysis (SGFA) for analyzing qMAP-PD neuroimaging data to identify Parkinson's disease subtypes.

Key Features
------------

* **Multi-modal Analysis**: Supports multiple neuroimaging modalities
* **Bayesian Inference**: Uses NumPyro/JAX for scalable MCMC
* **Clinical Validation**: Comprehensive clinical validation framework
* **Performance Optimized**: Memory-efficient processing for large datasets

Quick Start
-----------

.. code-block:: python

   from data.qmap_pd import load_qmap_pd_data
   from models.factory import create_model
   from analysis.model_runner import ModelRunner

   # Load data
   X_list = load_qmap_pd_data()
   
   # Create model
   model = create_model('sparse_gfa', config={'K': 10})
   
   # Run inference
   runner = ModelRunner(model)
   results = runner.run_mcmc(X_list)

Installation
------------

.. code-block:: bash

   pip install -r requirements.txt

Research Applications
--------------------

* **PD Subtyping**: Identify distinct Parkinson's disease subtypes
* **Biomarker Discovery**: Find neuroimaging biomarkers for disease progression  
* **Clinical Prediction**: Predict clinical outcomes from imaging data
* **Method Comparison**: Compare different factor analysis approaches

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`