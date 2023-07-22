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
This code is used to generate maps of mean Eccentricity predictions versus
mean measured/empirical values, for all participants in the NYU test set. 
The plots can be generated for both non-finetuned NYU test sets, and NYU test 
sets where the model was finetuned after training.
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
hemisphere = 'Left'

'''
How many participants were allocated to a 'Training' set for finetuning?
If num_finetuning_subjects == None, finetuning was not performed.
'''
num_finetuning_subjects = None
'''
How many epochs did finetuning occur for? If num_finetuning_subjects == None,
the value of num_epochs is ignored (as finetuning didn't take place).
'''
num_epochs = 20
'''
For which model (models 1-5) will predictions be plotted?
'''
selected_model = 5

# Create the file name components for the chosen prediction params
HEMI_FILENAME = hemisphere[0]
# Add additional info to filenames if finetuning is being used
FT_FILENAME = ""
if num_finetuning_subjects is not None:
    # Add the number of subjects used to finetune and number of epochs
    FT_FILENAME = \
        f'_finetuned_{num_finetuning_subjects}subj_{num_epochs}epochs'


# Total number of cortical nodes in the mesh
NUMBER_CORTICAL_NODES = int(64984)
# Number of nodes within each hemisphere
NUMBER_HEMI_NODES = int(NUMBER_CORTICAL_NODES / 2)

# Configure filepaths
sys.path.append('../..')
# For loading participants' curvature data
path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 
                'Retinotopy/data/nyu_converted')


# Loading participant IDs (in the order of selection for train/finetuning set and test set)
with open(osp.join(path, '../..', 'NYU_participant_IDs_in_order.txt')) as fp:
    subj = fp.read().split("\n")
subj = subj[0:len(subj) - 1]
'''
Get the ID of the last participant in the NYU Test set. This is the 43rd 
participant (stored at index 42) in the list of participant IDs in order. 
The curvature data for this participant is used as a background on the 
plotted surface.
'''
last_subj = subj[-1]

#### Loading curvature data for the last subject in the test set ####
curv_data = nib.load(osp.join(path, f'sub-wlsubj{last_subj}', 
    f'sub-wlsubj{last_subj}.curv.{HEMI_FILENAME.lower()}h.' +
    '32k_fs_LR.func.gii'))
curv_data = torch.tensor(np.reshape(curv_data.agg_data()
    .reshape((NUMBER_HEMI_NODES)), (-1, 1)), dtype=torch.float)
# Invert curvature values to resemble format of values in HCP dataset
curv_data *= -1

# Set the curvature background map
background = np.array(np.reshape(
                        curv_data.detach().numpy()[0:NUMBER_HEMI_NODES], (-1)))

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

# Set the name of the testset results directory
testset_results_dir = 'NYU_testset_results'
if num_finetuning_subjects is not None:
    # Add 'finetuned' to the testset results dir name if required
    testset_results_dir = 'NYU_testset_finetuned_results'

# Storing predicted and measured/empirical eccentricity values
pred = np.zeros((NUMBER_HEMI_NODES, 1))
measured = np.zeros((NUMBER_HEMI_NODES, 1))

# Load ECC predictions and measured values
predictions = torch.load(osp.join('./../..', 'Models', 'generalizability', 
testset_results_dir, f'NYU_testset{FT_FILENAME}-intactData_ECC_' + 
f'{HEMI_FILENAME}H_model{str(selected_model)}.pt'), map_location='cpu')

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
pred[final_mask == 1] = np.mean(pred_values, 0)
measured[final_mask == 1] = np.mean(measured_values, 0)

# Scaling
pred = np.array(pred) * 10 + threshold
measured = np.array(measured) * 10 + threshold

# Masking
measured[final_mask != 1] = 0
pred[final_mask != 1] = 0

#### Plot the mean test set predictions for the chosen model ####
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                    'Retinotopy/data/raw/surfaces',
                    f'S1200_7T_Retinotopy181.{HEMI_FILENAME}' +
                    '.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(pred[0:NUMBER_HEMI_NODES], (-1)), bg_map=background,
    cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=130,
    title=f'(NYU) Eccentricity {HEMI_FILENAME} hemisphere predictions - Model {selected_model} (test set mean)')
view.open_in_browser()

#### Plot the mean empirical test set data ####
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                    'Retinotopy/data/raw/surfaces',
                    f'S1200_7T_Retinotopy181.{HEMI_FILENAME}' +
                    '.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(measured[0:NUMBER_HEMI_NODES], (-1)), bg_map=background,
    cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=130,
    title=f'(NYU) Polar angle {HEMI_FILENAME} hemisphere ground truth (test set mean)')
view.open_in_browser()