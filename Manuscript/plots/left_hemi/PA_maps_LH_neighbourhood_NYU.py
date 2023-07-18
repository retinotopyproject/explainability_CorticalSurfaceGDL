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
This code is used to generate maps of Polar Angle predictions versus
measured/empirical values, for an NYU dataset participant from the test set. 
The plots can be generated for both non-finetuned NYU test sets, and NYU test 
sets where the model was finetuned after training. Plots for either the 
Left or Right hemisphere can be created.

Note: code implementation assumes that the file is being run from the dir 
explainability_CorticalSurfaceGDL/Manuscript/plots/left_hemi - I have modified 
the code to automatically set the working dir to this (if it isn't already).
"""
# Set the working directory to Manuscript/plots/left_hemi
os.chdir(osp.join(osp.dirname(osp.realpath(__file__))))

#### Params for selecting a model and a participant to plot ####
# Which hemisphere will predictions be graphed for? ('Left'/'Right')
hemisphere = 'Right'

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
For which anonymised participant in the test set will graphs be generated for?
eg. if participant_index == 0, graphs are generated for the 1st participant
in the test set.
'''
participant_index = 0
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
path = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 
                'Retinotopy/data/nyu_converted')


# Loading participant IDs (in the order of selection for train/finetuning set and test set)
with open(osp.join(path, '../..', 'NYU_participant_IDs_in_order.txt')) as fp:
    subj = fp.read().split("\n")
subj = subj[0:len(subj) - 1]


# Get the index of the first participant in the test set
test_index_start = 0
if num_finetuning_subjects is not None:
    # If finetuning, the test set
    test_index_start += num_finetuning_subjects
#### Loading curvature data for test set participants ####
curv = []
for index in range(test_index_start, len(subj)):
    curv_data = nib.load(osp.join(path, f'sub-wlsubj{subj[index]}', 
        f'sub-wlsubj{subj[index]}.curv.{HEMI_FILENAME.lower()}h.' +
        '32k_fs_LR.func.gii'))
    curv_data = torch.tensor(np.reshape(curv_data.agg_data()
        .reshape((NUMBER_HEMI_NODES)), (-1, 1)), dtype=torch.float)
    # Invert curvature values to resemble format of values in HCP dataset
    curv_data *= -1
    # Add curvature data to list of all participants' data
    curv.append(curv_data)

# Set the background curvature map
background = np.array(np.reshape(curv[participant_index][0:NUMBER_HEMI_NODES], (-1)))

threshold = 1  # Threshold for the curvature map

# Remove NaNs from curvature map
nocurv = np.isnan(background)
background[nocurv == 1] = 0
# Background settings (discretize curvature values to give a 2 colour map)
background[background < 0] = 0
background[background > 0] = 1

# Selecting all visual areas (Wang2015) plus V1-3 fovea
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


#### Loading Polar Angle data for the chosen participant ####

# Set the name of the testset results directory
testset_results_dir = 'NYU_testset_results'
if num_finetuning_subjects is not None:
    # Add 'finetuned' to the testset results dir name if required
    testset_results_dir = 'NYU_testset_finetuned_results'

# Set the colour maps used by the surf plots, for the given hemisphere
if hemisphere == 'Left':
    cmap = 'gist_rainbow_r'
else:
    # hemisphere == 'Right'
    cmap = 'gist_rainbow'


# Load PA data and create plots for the specified model

# Storing predicted and measured/empirical polar angle values
pred = np.zeros((NUMBER_HEMI_NODES, 1))
measured = np.zeros((NUMBER_HEMI_NODES, 1))

# Load PA predictions and measured values
predictions = torch.load(osp.join('./../../..', 'Models', 'generalizability', 
testset_results_dir, f'NYU_testset{FT_FILENAME}-intactData_PA_' + 
f'{HEMI_FILENAME}H_model{str(selected_model)}.pt'), map_location='cpu')

# Apply ROI mask to predicted and measured values
pred[final_mask_ROI == 1] = np.reshape(
    np.array(predictions['Predicted_values'][participant_index]),
    (-1, 1))
measured[final_mask_ROI == 1] = np.reshape(
    np.array(predictions['Measured_values'][participant_index]),
    (-1, 1))

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
measured[final_mask_ROI != 1] = 0 # Removes data from outside the ROI
# Not really sure what these masks do?
pred[pred == 1] = 100
pred[pred == 2] = 2
pred[final_mask_ROI != 1] = 0 # Removes data from outside the ROI


#### Plot the predictions for the chosen model ####
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                    'Retinotopy/data/raw/surfaces',
                    f'S1200_7T_Retinotopy181.{HEMI_FILENAME}' +
                    '.sphere.32k_fs_LR.surf.gii'),
    bg_map=background, surf_map=np.reshape(pred[0:NUMBER_HEMI_NODES], (-1)),
    cmap=cmap, black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=361,
    title=f'(NYU) Participant {participant_index+1}: Polar angle {HEMI_FILENAME} hemisphere predictions (Model {selected_model})')
view.open_in_browser()
# view.save_as_html(osp.join('D:\\Retinotopy Project\\surf_images\\NYU\\PA_LH\\', f'predicted_model{selected_model}_participant{participant_index+1}'))

#### Plot the empirical data ####
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                    'Retinotopy/data/raw/surfaces',
                    f'S1200_7T_Retinotopy181.{HEMI_FILENAME}' +
                    '.sphere.32k_fs_LR.surf.gii'),
    bg_map=background, surf_map=np.reshape(measured[0:NUMBER_HEMI_NODES], (-1)),
    cmap=cmap, black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=361,
    title=f'(NYU) Participant {participant_index+1}: Polar angle {HEMI_FILENAME} hemisphere ground truth')
view.open_in_browser()
# view.save_as_html(f'D:\\Retinotopy Project\\surf_images\\NYU\\PA_LH\\empirical_participant{participant_index+1}')