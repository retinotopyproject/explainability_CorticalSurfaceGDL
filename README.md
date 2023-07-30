# An explainability framework for cortical surface-based deep learning

This repository contains all source code necessary to replicate our recent work entitled "An explainability framework 
for cortical surface-based deep learning" available on [arXiv](https://arxiv.org/abs/2203.08312). Note that, this repo is a modified version of 
[deepRetinotopy](https://github.com/Puckett-Lab/deepRetinotopy).

## Table of Contents
* [Updates in this fork](#updates-in-this-fork)
* [Installation and requirements](#installation-and-requirements)
* [Explainability](#explainability)
* [Manuscript](#manuscript)
* [Models](#models)
* [Retinotopy](#retinotopy)
* [Citation](#citation)
* [Contact](#contact)

## Updates in this fork

New files were created to train and test models, using different data to the original project
and less input features, in order to test the models' generalizability to new datasets
and performance with only a single explicit input feature.
Models were created that used only curvature (no myelination data) as an input feature.
These were trained on participant data from the Human Connectome Project (HCP) dataset,
but with a more standard pre-processing pipeline applied to the data (instead of the
HCP-specific pre-processing pipeline that was previously used).

HCP dataset: https://balsa.wustl.edu/study/show/9Zkk
NYU retinotopic dataset: https://openneuro.org/datasets/ds003787/versions/1.0.0

Initially, HCP standard-processing pipeline models were trained on 161 participants, 
then evaluated with a development set and test set of 10 HCP participants each.
43 participants from the NYU retinotopic dataset were then used in a test set on
the HCP trained model, to test the generalizability of the pre-trained model on
unseen data from different datasets.
As well as this, this fork also explored finetuning the pre-trained HCP model on
a small selection of NYU participant data (8 or 12 participants), with the remaining
NYU participants being used in a test set.

## Installation and requirements 

Models were generated using [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/). Since this package 
is under constant updates, we highly recommend that 
you run the models in one of the following ways:

### Option 1: Neurodesktop
Neurodesktop is a browser-accessible, reproducible neuroimaging environment that
includes the 'deepretinotopy_1.0.1' container as a module. The code for this project
can be run within Neurodesktop by loading this container and running the code within it.

For more information on setting up Neurodesktop, including
helpful tutorials, see here: https://www.neurodesk.org/docs/getting-started/neurodesktop/

Neurodesk Play and Neurodesk Lab are recommended for quickly and easily trying out Neurodesktop
straight from the browser. For more info, see here: https://www.neurodesk.org/docs/getting-started/neurodesktop/play/

To launch the container in Neurodesktop: open the menu from the bottom leftmost icon the task bar >
go to Neurodesk > Machine Learning > deepretinotopy > deepretinotopy 1.0.1.
This will launch the container in a new terminal window, in which the project code can be ran.
Alternatively, run this command in the terminal:
```bash
	/bin/bash -i /neurocommand/local/bin/deepretinotopy-1_0_1.sh
```
This repository can be directly cloned into a Neurodesktop instance.
Before running any of the code, make sure to change the PYTHONPATH to the project's working 
directory by running this command inside the container:
```bash
	export PYTHONPATH=/working/dir
```

A local instance of VS Code can also be used to remotely connect to and run code
on Neurodesktop - see here: https://www.neurodesk.org/docs/getting-started/neurocommand/visual-studio-code/

### Option 2: Run locally (Docker container or Singularity image)
The Neurodesk 'neurocontainers' project contains a script for building a Docker
container for this project. The most recent version 'deepretinotopy_1.0.1'
can be found here: https://github.com/NeuroDesk/neurocontainers/tree/master/recipes/deepretinotopy

The container is built automatically, and hosted on dockerhub and github. To pull the container,
run this command in a terminal:
```bash
	docker pull vnmd/deepretinotopy_1.0.1
```
The container can be run interactively, and the code can be run from a local location outside of
the container (using a bind mount). As well as this, git is available within the container;
this repository can be directly cloned into an appropriate dir inside the container.
Before running any of the code, make sure to change the PYTHONPATH to the project's working 
directory by running this command inside the container:
```bash
	export PYTHONPATH=/working/dir
```
For more information see: https://www.neurodesk.org/docs/getting-started/neurocontainers/docker/

A singularity container (converted from the Docker container) can also be used.
For more detailed info/instructions for Singularity images and the Transparent Singularity Tool,
see here: https://www.neurodesk.org/docs/getting-started/neurocontainers/singularity/

Note: you may need to add the project's working directory to your device's PYTHONPATH.

### Option 3: Run locally (Conda environment)

- Create a new Conda environment with Python version 3.7.15.
- Activate the environment.
- Install torch first:

```bash
	conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

- Install torch-geometric and some other required packages:
```bash
	pip install packaging pandas seaborn nibabel torch-geometric==1.6.3 scikit-learn==0.22.2 scipy==1.1.0 matplotlib==3.3.0
```
	
- Install torch-scatter, torch-sparse, torch-cluster, and torch-spline-conv (run these commands in this order):
	 
```bash
	pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
	pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
	pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
	pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
```

Note, there are installations for different CUDA versions. For more: [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

- Finally, install the following git repository for plots:

```bash
    pip install git+https://github.com/felenitaribeiro/nilearn.git
```

Note: when running the project code locally, you may need to add the project's working directory to your device's PYTHONPATH.

## Explainability
This folder contains functions for the occlusion of input features within a target vertex's neighborhood.

## Manuscript

This folder contains all source code necessary to reproduce all figures and summary statistics in our manuscript.

Update: Scripts were added to generate appropriate graphs/plots/metrics for HCP standard-processing
trained models, with either HCP or NYU training set data. Figures could also be generated for
models finetuned with NYU data.

## Models

This folder contains all source code necessary to train a new model and to generate predictions on the test dataset 
using our pre-trained models. Note, models were updated for PyTorch 1.6.0. 

Update: New files can be used to train HCP models using standard-processing pipeline
data, finetune pre-trained models with NYU data, and generate development and test set
predictions for HCP standard-processing or NYU participants.

## Retinotopy

This folder contains all source code necessary to replicate datasets generation, in addition to functions and labels 
used for figures and models' evaluation. 

Update: Includes files for reading and processing both HCP data with a standard-processing pipeline 
applied, and NYU data.

## Citation

Please cite our papers if you used our model or if it was somewhat helpful for you :wink:

	@article{Ribeiro2022,
		author = {Ribeiro, Fernanda L and Bollmann, Steffen and Cunnington, Ross and Puckett, Alexander M},
		arxivId = {2203.08312},
		journal = {arXiv},
		keywords = {Geometric deep learning, high-resolution fMRI, vision, retinotopy, explainable AI},
		title = {{An explainability framework for cortical surface-based deep learning}},
		url = {https://arxiv.org/abs/2203.08312},
		year = {2022}
	}
	

	@article{Ribeiro2021,
		author = {Ribeiro, Fernanda L and Bollmann, Steffen and Puckett, Alexander M},
		doi = {https://doi.org/10.1016/j.neuroimage.2021.118624},
		issn = {1053-8119},
		journal = {NeuroImage},
		keywords = {cortical surface, high-resolution fMRI, machine learning, manifold, visual hierarchy,Vision},
		pages = {118624},
		title = {{Predicting the retinotopic organization of human visual cortex from anatomy using geometric deep learning}},
		url = {https://www.sciencedirect.com/science/article/pii/S1053811921008971},
		year = {2021}
	}


## Contact
Fernanda Ribeiro <[fernanda.ribeiro@uq.edu.au](fernanda.ribeiro@uq.edu.au)>
