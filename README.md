# Sparse Group Factor Analysis (GFA) for Parkinson's Subtyping

Python implementation of sparse GFA that can be used to identify latent disease factors that capture associations between various data modalities differently expressed within population subgroups. This project applies and adapts this model to the qMAP-PD (https://qmaplab.com/qmap-pd) study dataset.

## Description of the files:
- [get_data.py](get_data.py): script to load datasets or generate synthetic data from multiple data sources.
- [visualization.py](visualization.py): visualize the results of the experiments with real/synthetic data.
- [run_analysis.py](run_analysis.py): script that contains the sparse GFA model and is used to run the experiments. 
- [utils.py](utils.py): script that contains functions to support the run_analyis.py file.
- [loader_qmap_pd.py](loader_qmap_pd.py): loads and pre-processes qMAP-PD dataset, outputting multi-view datasets prepared for modeling.

## Installation
- Clone the repository.
- Create and activate a virtual environment.
- Install the necessary packages by running:
```
    pip install -r requirements.txt
```
## Running on Colab (GPU runtime)
- Open folder in Colab
- Select Runtime >> Change runtime type and select one of the GPU hardware accelerators
- Install dependencies
```
    !pip install -r requirements-colab.txt
```
- Install GPU-enabled JAX and NumPyro
```
    !pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    !pip install numpyro
```
- Add flag ```--device gpu``` when calling ```run_analysis.py```