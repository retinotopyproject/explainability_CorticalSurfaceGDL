import os.path as osp
import os
import sys
import numpy as np
import scipy.io
import torch
import nibabel as nib
from nilearn import plotting

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from Retinotopy.functions.error_metrics import smallest_angle


"""
This code was taken from the deepRetinotopy repository, from the file 
'SuppFigure5a_DeltaThetaVisualCortex_PA.py' in the Manuscript/plots/left_hemi
dir (https://github.com/Puckett-Lab/deepRetinotopy/)

The code generates a mean error and individual variability map for the 
HCP test set. This file has been modified, such that it can generate plots for
HCP data with a more standard pre-processsing pipeline applied (not the 
HCP-specific pre-processing pipeline). Error measures the difference between
predicted and empirical angles within each participant's data - lower error 
values are preferred, as these may indicate a stronger similarity between
the prediction and ground truth. Individual variability measures the 
differences between the predicted maps across all test set participants. 
As individual variability can indicate that the model may more successfully
replicate the individual-specific retinotopic characteristics of a participant, 
higher individual variability values are desired.

A map of the last test set participants' curvature data will be used as a 
background to the mean error and individual variability maps on the 
plotted surface.
Maps are generated per hemisphere (either Left or Right hemisphere), and can
be generated for any PA model (models 1-5).

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

# Loading participant IDs (in the order of selection for train/finetuning set and test set)
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

threshold = 1 # Threshold for the curvature map

# Remove NaNs from curvature map
nocurv = np.isnan(background)
background[nocurv == 1] = 0
# Background settings (discretize curvature values to give a 2 colour map)
background[background < 0] = 0
background[background > 0] = 1

mean_delta = [] # Prediction error
mean_across = [] # Individual variability

# Storing theta values
theta_withinsubj = []
theta_acrosssubj_pred = []

# Selecting all visual areas (Wang2015) plus V1-3 fovea
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
ROI1 = np.zeros((NUMBER_HEMI_NODES, 1))
# Create ROI mask for the relevant hemisphere
if hemisphere == 'Left':
    final_mask = final_mask_L
else:
    # hemisphere == 'Right'
    final_mask = final_mask_R
ROI1[final_mask == 1] = 1
mask = ROI1
mask = mask[ROI1 == 1]


#### Loading Polar Angle data for test set participants ####

# Load PA predictions and measured values
predictions = torch.load(osp.join('./../..','Models', 'generalizability', 
    'testset_results', 
    f'testset-intactData_PA_{HEMI_FILENAME}H_model{str(selected_model)}.pt'),
    map_location='cpu')

# Compute angle between predicted and empirical predictions across participants
for j in range(len(predictions['Predicted_values'])):
    theta_pred_across_temp = []

    for i in range(len(predictions['Predicted_values'])):
        '''
        Compute the difference between predicted and empirical angles
        within participant data (the error - ground truth vs prediction).
        '''
        if i == j:
            # Loading predicted values
            pred = np.reshape(np.array(predictions['Predicted_values'][i]),
                                (-1, 1))
            # Loading empirical values
            measured = np.reshape(
                np.array(predictions['Measured_values'][j]),
                (-1, 1))

            # Rescaling PA values to match the correct visual field
            minus = pred > 180
            sum = pred < 180
            pred[minus] = pred[minus] - 180
            pred[sum] = pred[sum] + 180
            # Convert from degrees to radians
            pred = np.array(pred) * (np.pi / 180)

            minus = measured > 180
            sum = measured < 180
            measured[minus] = measured[minus] - 180
            measured[sum] = measured[sum] + 180
            # Convert from degrees to radians
            measured = np.array(measured) * (np.pi / 180)

            '''
            Computing delta theta (the difference between the predicted and
            empirical angles).
            '''
            theta = smallest_angle(pred, measured)
            theta_withinsubj.append(theta)

        if i != j:
            '''
            Compute the difference between the predicted maps across all
            test set participants (the individual variability).
            '''

            # Loading predicted values
            pred = np.reshape(np.array(predictions['Predicted_values'][i]),
                                (-1, 1))
            pred2 = np.reshape(
                np.array(predictions['Predicted_values'][j]), (-1, 1))

            # Rescaling PA values to match the correct visual field
            minus = pred > 180
            sum = pred < 180
            pred[minus] = pred[minus] - 180
            pred[sum] = pred[sum] + 180
            # Convert from degrees to radians
            pred = np.array(pred) * (np.pi / 180)

            minus = pred2 > 180
            sum = pred2 < 180
            pred2[minus] = pred2[minus] - 180
            pred2[sum] = pred2[sum] + 180
            # Convert from degrees to radians
            pred2 = np.array(pred2) * (np.pi / 180)

            # Computing delta theta (difference between the predicted maps)
            theta_pred = smallest_angle(pred, pred2)
            theta_pred_across_temp.append(theta_pred)

    theta_acrosssubj_pred.append(np.mean(theta_pred_across_temp, axis=0))

# Computing the means:
mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)
mean_theta_acrosssubj_pred = np.mean(np.array(theta_acrosssubj_pred),
                                        axis=0)

mean_delta.append(mean_theta_withinsubj[mask == 1])
mean_across.append(mean_theta_acrosssubj_pred[mask == 1])

mean_delta = np.reshape(np.array(mean_delta), (1, -1))
mean_across = np.reshape(np.array(mean_across), (1, -1))

'''
Choose the model at index 0 in mean_delta and mean_across.
In this current implementation, mean_delta and mean_across will only contain
one model at index 0. However, the implementation can be modified to graph
and calculate error and individual variability for multiple models at once.
Implementation of this file in the deepRetinotopy repo added 4 different
model types (eg. 'pred' - intact features, rotated ROI, shuffled myelin and 
curvature values, constant values) to mean_delta and mean_across.
A similar implementaton could also potentially be used to graph error and
individual variability for all models 1-5 generated for the same hemisphere.
'''
model_index = 0

# Apply masks to remove data from outside ROI
delta_theta = np.ones((NUMBER_HEMI_NODES, 1))

delta_theta[final_mask == 1] = np.reshape(mean_delta[model_index],
                                (np.shape(mean_delta[model_index])[0], 1)) + threshold
delta_theta[final_mask != 1] = 0

delta_across = np.ones((NUMBER_HEMI_NODES, 1))

delta_across[final_mask == 1] = np.reshape(mean_across[model_index],
                                (np.shape(mean_delta[model_index])[0], 1)) + threshold
delta_across[final_mask != 1] = 0


#### Plot the error map ####
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                    'Retinotopy/data/raw/surfaces',
                    f'S1200_7T_Retinotopy181.{HEMI_FILENAME}' +
                    '.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(delta_theta[0:NUMBER_HEMI_NODES], (-1)), bg_map=background,
    cmap='Reds', black_bg=False, symmetric_cmap=False, threshold=threshold,
    vmax=75 + threshold,
    title=f'Prediction error - PA {hemisphere} hemisphere test set')
view.open_in_browser()

#### Plot the individual variability map ####
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                    'Retinotopy/data/raw/surfaces',
                    f'S1200_7T_Retinotopy181.{HEMI_FILENAME}' +
                    '.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(delta_across[0:NUMBER_HEMI_NODES], (-1)), bg_map=background,
    cmap='Blues', black_bg=False, symmetric_cmap=False, threshold=threshold,
    vmax=75 + threshold,
    title=f'Individual variability - PA {hemisphere} hemisphere test set')
view.open_in_browser()