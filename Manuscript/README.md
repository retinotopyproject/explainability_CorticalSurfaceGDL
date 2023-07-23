# Manuscript


This folder contains all source code necessary to reproduce all figures and summary statistics in our recent work entitled "An explainability framework 
for cortical surface-based deep learning" available on [arXiv](https://arxiv.org/abs/2203.08312).

## Figures

Figures were generated using the scripts in `./plots/left_hemi`.

## Descriptive statistics

Saliency maps were generated using the following scripts:
- ./stats/left_hemi/MeanDeltaTheta_vertices_PA_neighbor.py
- ./stats/left_hemi/MeanDeltaTheta_vertices_PA_neighbor_feat.py


### Changes in this project fork:
- Manuscript/stats/ModelEval_MeanDeltaTheta_PA.py and 
ModelEval_MeanDeltaTheta_ECC.py: Modified versions of files from the deepRetinotopy
repo (https://github.com/Puckett-Lab/deepRetinotopy/). Create various 
'ErrorPerParticipant' .npz files for the PA/ECC LH/RH Test set, which are used 
by Figure7b_DeltaTheta_ModelvsAverageMap.py. Can be used with HCP standard-processing
models, or NYU finetuned/non-finetuned models.
- Models/plots/correlation_plot.py: Used to measure the Pearson correlation
coefficient for each HCP participant's curvature data. Correlations between
the HCP-specific processing-pipeline and the HCP standard processing pipeline
were measured and graphed.
- Models/plots/curvature_plots.py: Can generate LH/RH curvature maps for any 
participant from the HCP-specific processing pipeline data, HCP 
standard-processing pipeline data, or the NYU dataset.
- Models/plots/Figure6R2Average_plot.py: Modified version of a file from the
deepRetinotopy repo. Used to measure the mean explained variance (R2 values)
for HCP test set participants (with a standard processing pipeline applied to
HCP data).
- Models/plots/Figure7b_DeltaTheta_ModelvsAverageMap.py: Taken from deepRetinotopy
repo. Modified to create prediction error point plots for HCP standard-processing
models, and NYU finetuned/non-finetuned models. This file requires that other
files are generated in SuppFigure3 PA/ECC files as well as the ModelEval_MeanDeltaTheta
PA/ECC files found in Manuscript/stats. See the docstrings in these files for more
details.
- Models/plots/HCP_stdprocessing_ECC_maps.py and HCP_stdprocessing_PA_maps_neighbourhood.py:
Used to generate comparison maps of a test set participant's model predictions
vs. ground truth data. Code was adapted from files such as PA_maps_LH_neighbourhood.py 
and modified to be used with HCP standard processing pipeline data.
- Models/plots/NYU_ECC_maps.py and NYU_PA_maps_neighbourhood.py: Similar to the 
previous files listed above - the NYU files create plots for NYU finetuned and 
non-finetuned models.
- Models/plots/HCP_stdprocessing_PA_maps_mean_testset.py, HCP_stdprocessing_ECC_maps_mean_testset.py,
NYU_PA_maps_mean_testset.py, NYU_ECC_maps_mean_testset.py: Similar to the previously
mentioned files, but used to create plots of mean test set model predictions vs
mean test set empirical data (for HCP standard-processing and NYU finetuned/
non-finetuned models).
- Models/plots/SuppFigure3_ECCaverage_plot.py, SuppFigure3_PAaverage_plot.py:
Generate mean training set empirical data maps for ECC/PA LH/RH, for HCP models
with a standard pre-processing pipeline applied.
- Models/plots/NYU_SuppFigure3_ECCaverage_plot.py, NYU_SuppFigure3_PAaverage_plot.py:
Similar to the above files, but used specifically only for NYU finetuned models.
If an NYU model is not finetuned, then its empirical mean training set data is
identical to the HCP map produced by SuppFigure3ECCaverage_plot.py and/or
SuppFigure3PAaverage_plot.py. If an NYU model is finetuned, then the NYU files
generate a mean empirical training set consisting only of the participants 
reserved for finetuning.
- SuppFigure5a_DeltaThetaVisualCortex_PA.py and NYU_SuppFigure5a_DeltaThetaVisualCortex_PA.py:
Generate mean error and individual variability test set maps for the HCP 
standard-processing or NYU finetuned/non-finetuned models.



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