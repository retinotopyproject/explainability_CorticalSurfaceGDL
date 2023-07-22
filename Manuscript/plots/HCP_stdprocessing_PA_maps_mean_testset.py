import os
import os.path as osp
import sys
import numpy as np
import scipy.io
import torch
import nibabel as nib
from nilearn import plotting

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from Retinotopy.functions.def_ROIs_DorsalEarlyVisualCortex import roi as roi2

"""
This code is used to generate maps of mean Polar Angle predictions versus
mean measured/empirical values, for all participants in the HCP test set. This 
file is used for models using HCP data, with a more standardised pre-processing 
pipeline (different to the HCP-specific pre-processing pipeline) applied to
participant data.
Plots for either the Left or Right hemisphere can be created, for any of the
models generated during training (models 1-5).
A map of the last test set participant's curvature data will be used 
as a background to the mean PA map on the plotted surface.

Note: code implementation assumes that the file is being run from the dir 
explainability_CorticalSurfaceGDL/Manuscript/plots - I have modified 
the code to automatically set the working dir to this (if it isn't already).
"""
# Set the working directory to Manuscript/plots
os.chdir(osp.join(osp.dirname(osp.realpath(__file__))))

#### Params for selecting a model and a participant to plot ####
# Which hemisphere will predictions be graphed for? ('Left'/'Right')
hemisphere = 'Right'

'''
For which model (models 1-5) will mean predictions be plotted?
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
'''
Get the ID of the last participant in the HCP Test set. This is the 181st 
participant (stored at index 180) in the list of participant IDs in order. 
The curvature data for this participant is used as a background on the 
plotted surface.
'''
last_subj = subj[-1]

#### Loading curvature data for the last subject in the test set ####
curv_data = nib.load(osp.join(path, 'fs-curvature', f'{last_subj}', \
    f'{str(last_subj)}.{HEMI_FILENAME}.curvature.32k_fs_LR.shape.gii'))
curv_data = torch.tensor(np.reshape(
                curv_data.agg_data().reshape((NUMBER_HEMI_NODES)), (-1, 1)), 
                dtype=torch.float)

# Set the curvature background map
background = np.array(np.reshape(
                        curv_data.detach().numpy()[0:NUMBER_HEMI_NODES], (-1)))

threshold = 1  # Threshold for the curvature map

# Remove NaNs from curvature map
nocurv = np.isnan(background)
background[nocurv == 1] = 0
# Background settings (discretize curvature values to give a 2 colour map)
background[background < 0] = 0
background[background > 0] = 1

# Sekecting all visual areas (Wang2015) plus V1-3 fovea
label_primary_visual_areas = ['ROI']
final_mask_L_ROI, final_mask_R_ROI, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
ROI_masked = np.zeros((NUMBER_HEMI_NODES, 1))
# Apply ROI mask for the relevant hemisphere
if hemisphere == 'Left':
    final_mask_ROI = final_mask_L_ROI
else:
    # hemisphere == 'Right'
    final_mask_ROI = final_mask_R_ROI
ROI_masked[final_mask_ROI == 1] = 1

# Dorsal V1-V3
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(['ROI'])
dorsal_earlyVisualCortex = np.zeros((NUMBER_HEMI_NODES, 1))
# Apply V1-V3 mask for the relevant hemisphere
if hemisphere == 'Left':
    final_mask = final_mask_L
else:
    # hemisphere == 'Right'
    final_mask = final_mask_R
dorsal_earlyVisualCortex[final_mask == 1] = 1


# Set the colour maps used by the surf plots, for the given hemisphere
if hemisphere == 'Left':
    cmap = 'gist_rainbow_r'
else:
    # hemisphere == 'Right'
    cmap = 'gist_rainbow'


#### Loading Polar Angle data for test set participants ####

# Storing predicted and measured/empirical polar angle values
pred = np.zeros((NUMBER_HEMI_NODES, 1))
measured = np.zeros((NUMBER_HEMI_NODES, 1))

# Load PA predictions and measured values
predictions = torch.load(osp.join('./../..','Models', 'generalizability', 
    'testset_results', 
    f'testset-intactData_PA_{HEMI_FILENAME}H_model{str(selected_model)}.pt'),
    map_location='cpu')

pred_values = []
measured_values = []
for participant_index in range(0, len(predictions)):
    # Apply ROI mask to predicted and measured values
    pred_values.append(np.reshape(
        np.array(predictions['Predicted_values'][participant_index]),
        (-1, 1)))
    measured_values.append(np.reshape(
        np.array(predictions['Measured_values'][participant_index]),
        (-1, 1)))

# Calculate the mean predicted and empirical maps
pred[final_mask_ROI == 1] = np.mean(pred_values, 0)
measured[final_mask_ROI == 1] = np.mean(measured_values, 0)

# Translating predicted and measured Polar Angle values
pred = np.array(pred)
minus = pred > 180
sum = pred < 180
pred[minus] = pred[minus] - 180 + threshold
pred[sum] = pred[sum] + 180 + threshold
pred = np.array(pred)

measured = np.array(measured)
minus = measured > 180
sum = measured < 180
measured[minus] = measured[minus] - 180 + threshold
measured[sum] = measured[sum] + 180 + threshold
measured = np.array(measured)

# Masking
measured[final_mask_ROI != 1] = 0 # Removes data from outside ROI
# Not really sure what these masks do?
pred[pred == 1] = 100
pred[pred == 2] = 2
pred[final_mask_ROI != 1] = 0 # Removes data from outside ROI


#### Plot the mean test set predictions for the chosen model ####
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                    'Retinotopy/data/raw/surfaces',
                    f'S1200_7T_Retinotopy181.{HEMI_FILENAME}' +
                    '.sphere.32k_fs_LR.surf.gii'),
    bg_map=background, surf_map=np.reshape(pred[0:NUMBER_HEMI_NODES], (-1)),
    cmap=cmap, black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=361,
    title=f'Polar angle {HEMI_FILENAME} hemisphere predictions - Model {selected_model} (test set mean)')
view.open_in_browser()
# view.save_as_html(f'D:\Retinotopy Project\surf_images\PA_LH_mean_testset\predicted_model{selected_model}_mean')

#### Plot the mean empirical test set data ####
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                    'Retinotopy/data/raw/surfaces',
                    f'S1200_7T_Retinotopy181.{HEMI_FILENAME}' +
                    '.sphere.32k_fs_LR.surf.gii'),
    bg_map=background, surf_map=np.reshape(measured[0:NUMBER_HEMI_NODES], (-1)),
    cmap=cmap, black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=361,
    title=f'Polar angle {HEMI_FILENAME} hemisphere ground truth (test set mean)')
view.open_in_browser()
# view.save_as_html(f'D:\Retinotopy Project\surf_images\PA_LH_mean_testset\empirical_model_mean')

