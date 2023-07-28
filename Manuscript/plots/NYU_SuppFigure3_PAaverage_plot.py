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
'SuppFigure3_PAaverage_plot.py' in the Manuscript dir
(https://github.com/Puckett-Lab/deepRetinotopy/)

The code generates a mean Polar Angle map of observed (ground truth) values
for NYU participants added to the Training set for finetuning.
A map of the first participants' curvature data will be used as a background
to the mean PA map on the plotted surface.
Mean PA maps are generated per hemisphere (either Left or Right hemisphere).

For models that use NYU data but don't perform finetuning, the file 
SuppFigure3_PAaverage_plot.py should be run instead. As no NYU participants
are added to the training set when no finetuning occurs, the average PA
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
hemisphere = 'Left'

'''
How many participants were allocated to a 'Training' set for finetuning?
'''
num_finetuning_subjects = 12

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

threshold = 1  # Threshold for the curvature map

# Remove NaNs from curvature map
nocurv = np.isnan(background)
background[nocurv == 1] = 0
# Background settings (discretize curvature values to give a 2 colour map)
background[background < 0] = 0
background[background > 0] = 1


#### Loading Polar Angle data for all subjects in Train set ####

# A pre-transform to be applied to the data
pre_transform = T.Compose([T.FaceToEdge()])

# Load the Training set
train_dataset = Retinotopy(path, 'Train', transform=T.Cartesian(),
                                 pre_transform=pre_transform, 
                                 n_examples=N_EXAMPLES, prediction='polarAngle', 
                                 hemisphere=hemisphere,
                                 num_finetuning_subjects=num_finetuning_subjects)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Selecting all visual areas (Wang2015) plus V1-3 fovea
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)


#### Get the mean PA map ####

# Create an array of zeroes for PA map values
PolarAngle = np.zeros((NUMBER_HEMI_NODES, 1))

# Load all PA maps for each Train set participant
PA = []
for data in train_loader:
    PA.append(np.array(data.y))
# Calculate a mean map
PA = np.mean(PA, 0)

# Create an output folder if it doesn't already exist
directory = './output'
if not osp.exists(directory):
    os.makedirs(directory)

# Saving the average map
np.savez(f'./output/NYU{FT_FILENAME}_AveragePolarAngleMap_{HEMI_FILENAME}H.npz', 
            list=PA)


#### Settings for plot ####
if hemisphere == 'Left':
    PolarAngle[final_mask_L == 1] = np.reshape(PA, (-1, 1))
    # Translating Left hemisphere polar angle values
    PolarAngle = np.array(PolarAngle)
    minus = PolarAngle > 180
    sum = PolarAngle < 180
    PolarAngle[minus] = PolarAngle[minus] - 180
    PolarAngle[sum] = PolarAngle[sum] + 180
    # Masking
    PolarAngle[final_mask_L == 1] += threshold
    PolarAngle[final_mask_L != 1] = 0
    # Set the colour map for the Left hemi
    cmap = 'gist_rainbow_r'
else:
    # hemisphere == 'Right'
    PolarAngle[final_mask_R == 1] = np.reshape(PA, (-1, 1))
    # Translating Right hemisphere polar angle values
    PolarAngle = np.array(PolarAngle)
    minus = PolarAngle > 180
    sum = PolarAngle < 180
    PolarAngle[minus] = PolarAngle[minus] - 180
    PolarAngle[sum] = PolarAngle[sum] + 180
    # Masking
    PolarAngle[final_mask_R == 1] += threshold
    PolarAngle[final_mask_R != 1] = 0
    # Set the colour map for the Right hemi
    cmap = 'gist_rainbow'


#### Create the surf plot ####
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                       'Retinotopy/data/raw/surfaces' +
                       f'/S1200_7T_Retinotopy181.{HEMI_FILENAME}' +
                       '.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(PolarAngle[0:NUMBER_HEMI_NODES], (-1)), 
    bg_map=background, cmap=cmap, black_bg=False, 
    symmetric_cmap=False, threshold=threshold, vmax=361,
    title=f'NYU finetuned - Polar angle {hemisphere} hemisphere mean ground truth (test set)')

# Show in web browser
view.open_in_browser()

