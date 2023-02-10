import numpy as np
import scipy.io
import os.path as osp
import torch
import sys

sys.path.append('..')

from nilearn import plotting
from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi

# For loading new data
import nibabel as nib

# subject_index = 8

# hcp_id = ['617748', '191336', '572045', '725751', '198653',
#           '601127', '644246', '191841', '680957', '157336']

# path = './../../../Retinotopy/data/raw/converted'
path = './../../../Retinotopy/data/nyu_converted'

# curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
# background = np.reshape(
#     curv['x' + hcp_id[subject_index] + '_curvature'][0][0][0:32492], (-1))

# For NYU curvature data:
# Loading subject IDs (in the order in which they were selected for train, dev, test datasets)
with open(osp.join(path, '../..', 'participant_IDs_in_order.txt')) as fp:
    subjects = fp.read().split("\n")
subjects = subjects[0:len(subjects) - 1]

# Defining number of nodes
number_cortical_nodes = int(64984)
number_hemi_nodes = int(number_cortical_nodes / 2)

curv = []
# Index of the first subject in the testing dataset
test_index_start = 0
for index in range(test_index_start, len(subjects)):
    # Reading NYU curvature data (Left hemi)
    new_data = nib.load(osp.join(path, f'sub-wlsubj{subjects[index]}', 
        f'sub-wlsubj{subjects[index]}.curv.lh.32k_fs_LR.func.gii'))
    new_data = torch.tensor(np.reshape(new_data.agg_data()
        .reshape((number_hemi_nodes)), (-1, 1)), dtype=torch.float)
    # Invert curvature values to resemble format of values in HCP dataset
    new_data *= -1
    curv.append(new_data)
# background = np.reshape(curv[subject_index][0][0:32492], (-1))
subject_index = 10
# for subject_index in range(0, 10):
background = np.array(np.reshape(curv[subject_index][0:32492], (-1)))

threshold = 10  # threshold for the curvature map

# Background settings
nocurv = np.isnan(background)
background[nocurv == 1] = 0
# background[background < 0] = 0
# background[background > 0] = 1

# ROI settings
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)

pred = np.zeros((32492, 1))
measured = np.zeros((32492, 1))

selected_model = 1
# Loading the predictions
predictions = torch.load('./../../../Models/generalizability/NYU_testset_results/'
    '/NYU_testset-intactData_ECC_LH_model' + str(selected_model) + '.pt',
    map_location='cpu')

pred[final_mask_L == 1] = np.reshape(
    np.array(predictions['Predicted_values'][subject_index]),
    (-1, 1))
measured[final_mask_L == 1] = np.reshape(
    np.array(predictions['Measured_values'][subject_index]),
    (-1, 1))

# Scaling
pred = np.array(pred) * 10 + threshold
measured = np.array(measured) * 10 + threshold

# Masking
measured[final_mask_L != 1] = 0
pred[final_mask_L != 1] = 0

# Empirical map
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(measured[0:32492], (-1)), bg_map=background,
    cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=130,
    title=f'Participant {subject_index+1}: Eccentricity Left hemisphere ground truth')
view.open_in_browser()

# Predicted map
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(pred[0:32492], (-1)), bg_map=background,
    cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=130,
    title=f'Participant {subject_index+1}: Eccentricity Left hemisphere predictions (Model {selected_model})')
view.open_in_browser()