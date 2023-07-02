# Retinotopy

This folder contains all source code necessary to replicate datasets generation, in addition to functions and labels 
used for figures and models' evaluation. 


### Changes in this project fork:
- Retinotopy/dataset/HCP_stdprocessing_3sets_ROI.py: this file is a modified
version of HCP_3sets_ROI.py, used to generate the Train, Development, and Test
sets. This code was modified to read and generate sets for HCP data with a
more standard pre-processing pipeline applied (different to the HCP 
pre-processing pipeline), and with only curvature data used as a feature set
for the models.
- Retinotopy/dataset/NYU_3sets_ROI.py: this file is used to generate sets 
required for models using the NYU dataset. These models used NYU participants
in their Test set, with the model being trained on HCP data points with
a standard processing pipeline. In cases where the trained model was finetuned
with a small quantity of NYU data points, this code could also be used to
generate a 'Train' set of participants reserved for finetuning.
- Retinotopy/read/read_HCPdata_stdprocessing.py: a modified version of
read_HCPdata.py. Used to read data for HCP participants, but applying a more
standard pre-processing pipeline to their data, and using only curvature in the
model's feature set. The method read_HCP in this class was configured to read 
and process myelination and pRF size data; however, this information was not
used by the generated models.
- Retinotopy/read/read_NYUdata.py: reads data for NYU participants. The 
read_NYU method is not configured to load and process data such as myelination, 
pRF size, or R2 values. As such, any models using data from the NYU dataset
cannot use myelination data as an input feature.
- Different data sources were used for this project (NYU Retinotopic dataset,
HCP dataset with a standard pre-processing pipeline). The files for the HCP
data used in the original project (that were available on the original project's
repository) can be found in Retinotopy/data/old_data.


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

