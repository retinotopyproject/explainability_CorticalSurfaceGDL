import numpy as np
import scipy.io
import os
import os.path as osp
import torch
import sys
import nibabel as nib
from nilearn import plotting

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi

"""
This code was taken from the deepRetinotopy repository and modified, from the 
file 'Eccentricity_maps_LH.py' in the Manuscript/plots/left_hemi dir
(https://github.com/Puckett-Lab/deepRetinotopy/)

This code is used to generate maps of Eccentricity predictions versus
measured/empirical values, for an HCP dataset participant from the test set. 
This file has been modified to generate these maps 
for HCP data pre-processed using a standard processing pipeline (different to 
the HCP-specific processing pipeline). Plots for either the 
Left or Right hemisphere can be created.

Note: code implementation assumes that the file is being run from the dir 
explainability_CorticalSurfaceGDL/Manuscript/plots - I have modified 
the code to automatically set the working dir to this (if it isn't already).
"""
# Set the working directory to Manuscript/plots
os.chdir(osp.join(osp.dirname(osp.realpath(__file__))))

#### Params for selecting a model and a participant to plot ####
# Which hemisphere will predictions be graphed for? ('Left'/'Right')
hemisphere = 'Left'

'''
For which anonymised participant in the test set will graphs be generated for?
eg. if participant_index == 0, graphs are generated for the 1st participant
in the test set. As there are 10 test set participants, this number must be in
the range [0, 9].
'''
participant_index = 0
'''
For which model (models 1-5) will predictions be plotted?
'''
selected_model = 5

# Create the file name components for the chosen prediction params
HEMI_FILENAME = hemisphere[0]


# Total number of cortical nodes in the mesh
NUMBER_CORTICAL_NODES = int(64984)
# Number of nodes within each hemisphere
NUMBER_HEMI_NODES = int(NUMBER_CORTICAL_NODES / 2)

# Configure filepaths
sys.path.append('../..')
# For loading participants' curvature data
path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 
                'Retinotopy/data/raw/converted')


# Loading participant IDs (in the order of selection for train, dev, test datasets)
with open(osp.join(path, '../../..', 'participant_IDs_in_order.txt')) as fp:
    subj = fp.read().split("\n")
subj = subj[0:len(subj) - 1]


#### Loading curvature data for test set participants ####
curv = []
# Index of the first participant in the testing dataset
test_index_start = int(171)
for index in range(test_index_start, len(subj)):
    new_data = nib.load(osp.join(path, f'fs-curvature/{subj[index]}/', \
        f'{subj[index]}.{HEMI_FILENAME}.curvature.32k_fs_LR.shape.gii'))
    new_data = torch.tensor(np.reshape(
                            new_data.agg_data().reshape((NUMBER_HEMI_NODES)), 
                            (-1, 1)), dtype=torch.float)
    # Add curvature data to list of all participants' curv data
    curv.append(new_data)

# Set the background curvature map
background = np.array(np.reshape(curv[participant_index][0:NUMBER_HEMI_NODES], (-1)))

threshold = 10  # Threshold for the curvature map

# Remove NaNs from curvature map
nocurv = np.isnan(background)
background[nocurv == 1] = 0
# Background settings (discretize curvature values to give a 2 colour map)
background[background < 0] = 0
background[background > 0] = 1

# Selecting all visual areas (Wang2015) plus V1-3 fovea
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
# Set ROI mask for the relevant hemisphere
if hemisphere == 'Left':
    final_mask = final_mask_L
else:
    # hemisphere == 'Right'
    final_mask = final_mask_R


#### Loading eccentricity data for the chosen participant ####

# Storing predicted and measured/empirical eccentricity values
pred = np.zeros((NUMBER_HEMI_NODES, 1))
measured = np.zeros((NUMBER_HEMI_NODES, 1))

# Load PA predictions and measured values
predictions = torch.load(osp.join('./../..','Models', 'generalizability', 
    'testset_results', 
    f'testset-intactData_ECC_{HEMI_FILENAME}H_model{str(selected_model)}.pt'),
    map_location='cpu')

# Apply ROI mask to predicted and measured values
pred[final_mask == 1] = np.reshape(
    np.array(predictions['Predicted_values'][participant_index]),
    (-1, 1))
measured[final_mask == 1] = np.reshape(
    np.array(predictions['Measured_values'][participant_index]),
    (-1, 1))

# Scaling
pred = np.array(pred) * 10 + threshold
measured = np.array(measured) * 10 + threshold

# Masking
measured[final_mask != 1] = 0
pred[final_mask != 1] = 0

#### Plot the predictions for the chosen model ####
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                    'Retinotopy/data/raw/surfaces',
                    f'S1200_7T_Retinotopy181.{HEMI_FILENAME}' +
                    '.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(pred[0:NUMBER_HEMI_NODES], (-1)), bg_map=background,
    cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=130,
    title=f'Participant {participant_index+1}: Eccentricity Left hemisphere predictions (Model {selected_model})')
view.open_in_browser()

#### Plot the empirical data ####
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                    'Retinotopy/data/raw/surfaces',
                    f'S1200_7T_Retinotopy181.{HEMI_FILENAME}' +
                    '.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(measured[0:NUMBER_HEMI_NODES], (-1)), bg_map=background,
    cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=130,
    title=f'Participant {participant_index+1}: Eccentricity Left hemisphere ground truth')
view.open_in_browser()