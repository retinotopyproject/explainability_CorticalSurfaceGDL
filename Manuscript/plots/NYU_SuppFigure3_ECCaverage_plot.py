import os
import os.path as osp
import sys
import torch_geometric.transforms as T
import numpy as np
import scipy.io
import torch
import nibabel as nib
from nilearn import plotting
from torch_geometric.data import DataLoader

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from Retinotopy.dataset.NYU_3sets_ROI import Retinotopy


"""
This code was taken from the deepRetinotopy repository, from the file 
'SuppFigure3_ECCaverage_plot.py' in the Manuscript dir
(https://github.com/Puckett-Lab/deepRetinotopy/)

The code generates a mean Eccentricity map of observed (ground truth) values
for NYU participants added to the Training set for finetuning.
A map of the first participants' curvature data will be used as a background
to the mean ECC map on the plotted surface.
Mean ECC maps are generated per hemisphere (either Left or Right hemisphere).
This file is also used to generate the files of the naming format
'MaskEccentricity_above1below8ecc_{L or R}H.npz'.

For models that use NYU data but don't perform finetuning, the file 
SuppFigure3_ECCaverage_plot.py should be run instead. As no NYU participants
are added to the training set when no finetuning occurs, the average ECC
plot for the training set would be identical to the plot for models using
only HCP data.

Note: code implementation assumes that the file is being run from the dir 
Manuscript/plots - I have modified the code to automatically set the 
working dir to this (if it isn't already).
"""
# Set the working directory to Manuscript/plots
os.chdir(osp.dirname(osp.realpath(__file__)))

#### Params used for model predictions ####
# Which hemisphere will predictions be generated for? ('Left'/'Right')
hemisphere = 'Right'

'''
How many participants were allocated to a 'Training' set for finetuning?
'''
num_finetuning_subjects = 8

# Create the file name components for the chosen prediction params
HEMI_FILENAME = hemisphere[0]
# Add additional info to filenames if finetuning is being used
FT_FILENAME = ""
if num_finetuning_subjects is not None:
    # Add the number of subjects used to finetune and number of epochs
    FT_FILENAME = \
        f'_finetuned_{num_finetuning_subjects}subj'


# The number of participants (total) in all model sets
N_EXAMPLES = 43
# Total number of cortical nodes in the mesh
NUMBER_CORTICAL_NODES = int(64984)
# Number of nodes within each hemisphere
NUMBER_HEMI_NODES = int(NUMBER_CORTICAL_NODES / 2)


# Configure filepaths
sys.path.append('../..')
# For loading data from the Training set
path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'Retinotopy/data')
# For loading participants' curvature data
path_curv = osp.join(path, 'nyu_converted')

# Load participant IDs, in the order they were selected for train, dev, test sets
with open(osp.join(path, '..', 'NYU_participant_IDs_in_order.txt')) as fp:
    subj = fp.read().split("\n")
subj = subj[0:len(subj) - 1]
'''
Get the ID of the first participant in the NYU Training set. The curvature data 
for this participant is used as a background on the plotted surface.
'''
first_subj = subj[0]


#### Loading curvature data for the first subject in training set ####
curv_data = nib.load(osp.join(path_curv, f'sub-wlsubj{first_subj}', 
    f'sub-wlsubj{first_subj}.curv.{HEMI_FILENAME.lower()}h.32k_fs_LR.func.gii'))
curv_data = torch.tensor(np.reshape(curv_data.agg_data()
    .reshape((NUMBER_HEMI_NODES)), (-1, 1)), dtype=torch.float)
# Invert curvature values to resemble format of values in HCP dataset
curv_data *= -1 

# Set the background 
background = np.array(np.reshape(
                        curv_data.detach().numpy()[0:NUMBER_HEMI_NODES], (-1)))

threshold = 10  # Threshold for the curvature map

# Remove NaNs from curvature map
nocurv = np.isnan(background)
background[nocurv == 1] = 0
# Background settings (discretize curvature values to give a 2 colour map)
background[background < 0] = 0
background[background > 0] = 1


#### Loading Eccentricity data for all subjects in Train set ####

# A pre-transform to be applied to the data
pre_transform = T.Compose([T.FaceToEdge()])

# Load the Training set
train_dataset = Retinotopy(path, 'Train', transform=T.Cartesian(),
                                 pre_transform=pre_transform, 
                                 n_examples=N_EXAMPLES, 
                                 prediction='eccentricity', 
                                 hemisphere=hemisphere,
                                 num_finetuning_subjects=num_finetuning_subjects)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Selecting all visual areas (Wang2015) plus V1-3 fovea
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)


#### Get the mean ECC map ####

# Create an array of zeroes for ECC map values
ecc_thr = np.zeros((NUMBER_HEMI_NODES, 1))

# Load all ECC maps for each Train set participant
ecc = []
for data in train_loader:
    ecc.append(np.array(data.y))
# Calculate a mean map
ecc = np.mean(ecc, 0)

# Create an output folder if it doesn't already exist
directory = './output'
if not osp.exists(directory):
    osp.makedirs(directory)

# Saving the average map
np.savez(f'./output/NYU{FT_FILENAME}_AverageEccentricityMap_{HEMI_FILENAME}H.npz', 
            list=ecc)

# Create/save a mask for prediction errors
ecc_1to8 = []
for i in range(len(ecc)):
    if ecc[i][0] < 1 or ecc[i][0] > 8:
        ecc_1to8.append(0)
    else:
        ecc_1to8.append(ecc[i][0])
ecc_1to8 = np.reshape(np.array(ecc_1to8),(-1))
np.savez(f'./output/NYU{FT_FILENAME}_MaskEccentricity_above1below8ecc_{HEMI_FILENAME}H', 
        list = ecc_1to8 > 0)


#### Settings for plot ####

if hemisphere == 'Left':
    # Masking
    ecc_thr[final_mask_L == 1] = np.reshape(ecc, (-1, 1)) * 10 + threshold
    ecc_thr[final_mask_L != 1] = 0
else:
    # hemisphere == 'Right'
    # Masking
    ecc_thr[final_mask_R == 1] = np.reshape(ecc, (-1, 1)) * 10 + threshold
    ecc_thr[final_mask_R != 1] = 0


#### Create the surf plot ####
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                       'Retinotopy/data/raw/surfaces' +
                       f'/S1200_7T_Retinotopy181.{HEMI_FILENAME}' + 
                       '.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(ecc_thr[0:NUMBER_HEMI_NODES], (-1)), bg_map=background,
    cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False, 
    threshold=threshold, vmax=130,
    title=f'NYU finetuned - Eccentricity {hemisphere} hemisphere mean ground truth (test set)')

# Show in web browser
view.open_in_browser()
# Save plot as a HTML file
# view.save_as_html(f'D:\\Retinotopy Project\\surf_images\\ECC_{HEMI_FILENAME}H_mean_trainset\\NYU_empirical_mean_trainset_ECC_{HEMI_FILENAME}H')

