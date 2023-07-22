import os.path as osp
import os
import scipy.io
import sys
import torch_geometric.transforms as T
import numpy as np
import nibabel as nib
import torch
from nilearn import plotting
from torch_geometric.data import DataLoader

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from Retinotopy.dataset.HCP_stdprocessing_3sets_ROI import Retinotopy

"""
This code was taken from the deepRetinotopy repository, from the file 
'Figure6_R2Average_plot.py' in the Manuscript/plots/left_hemi dir
(https://github.com/Puckett-Lab/deepRetinotopy/)
The file generates an average explained variance (mean R2 values) map
for HCP test set participants. This file has been modified to generate 
these maps for HCP data pre-processed using a standard processing pipeline 
(different to the HCP-specific processing pipeline).
Mean R2 maps are generated per hemisphere (either Left or Right hemisphere).

Note: code implementation assumes that the file is being run from the dir 
explainability_CorticalSurfaceGDL/Manuscript/plots/left_hemi - I have modified 
the code to automatically set the working dir to this (if it isn't already).
"""
# Set the working directory to Manuscript/plots/left_hemi
os.chdir(osp.join(osp.dirname(osp.realpath(__file__))))

#### Params for selecting a model to plot ####
# Which hemisphere will predictions be graphed for? ('Left'/'Right')
hemisphere = 'Left'

# Create the file name components for the chosen prediction params
HEMI_FILENAME = hemisphere[0]


# The number of participants (total) in all model sets
N_EXAMPLES = 181
# Total number of cortical nodes in the mesh
NUMBER_CORTICAL_NODES = int(64984)
# Number of nodes within each hemisphere
NUMBER_HEMI_NODES = int(NUMBER_CORTICAL_NODES / 2)

# Configure filepaths
sys.path.append('../..')
# For loading data from the Training set
path = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'Retinotopy/data')
# For loading participants' curvature data
path_curv = osp.join(path, 'raw/converted')


# Loading participant IDs (in the order of selection for train, dev, test datasets)
with open(osp.join(path_curv, '../../..', 'participant_IDs_in_order.txt')) as fp:
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
curv_data = nib.load(osp.join(path_curv, 'fs-curvature', f'{last_subj}', \
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

# Sekecting all visual areas (Wang2015) plus V1-3 fovea
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
R2_thr = np.zeros((NUMBER_HEMI_NODES, 1))
# Select ROI mask for the relevant hemisphere
if hemisphere == 'Left':
    final_mask = final_mask_L
else:
    # hemisphere == 'Right'
    final_mask = final_mask_R


#### Loading R2 values for all test set participants ####

# A pre-transform to be applied to the data
pre_transform = T.Compose([T.FaceToEdge()])

# Loading data from the test set
test_dataset = Retinotopy(path, 'Test', transform=T.Cartesian(),
                           pre_transform=pre_transform, n_examples=N_EXAMPLES,
                           prediction='polarAngle', myelination=False,
                           hemisphere=hemisphere)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Load R2 values
R2 = []
for data in test_loader:
    R2.append(np.array(data.R2))
# Calculate the average explained variance map
R2 = np.mean(R2, 0)

# Masking
R2_thr[final_mask == 1] = np.reshape(R2, (-1, 1)) + threshold
R2_thr[final_mask != 1] = 0

#### Plot the explained variance map ####
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces' +
                       f'/S1200_7T_Retinotopy181.{HEMI_FILENAME}' +
                       '.sphere.32k_fs_LR.surf.gii'),
    bg_map=background, surf_map=np.reshape(R2_thr[0:NUMBER_HEMI_NODES], (-1)),
    threshold=threshold, cmap='hot', black_bg=False, symmetric_cmap=False,
    vmax=60 + threshold,
    title=f'{hemisphere} hemisphere mean explained variance (test set)')
view.open_in_browser()
# view.save_as_html(f'D:\\Retinotopy Project\\surf_images\\R2_average\\R2_average_LH')