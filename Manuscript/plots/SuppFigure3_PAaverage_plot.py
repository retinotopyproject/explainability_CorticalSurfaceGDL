import os.path as osp
import sys
import torch_geometric.transforms as T
import numpy as np
import scipy.io

# For loading new curvature data
import torch
import nibabel as nib

sys.path.append('../..')

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from nilearn import plotting
from Retinotopy.dataset.HCP_stdprocessing_3sets_ROI import Retinotopy
from torch_geometric.data import DataLoader


# subject_index = 7

# hcp_id = ['617748', '191336', '572045', '725751', '198653',
#           '601127', '644246', '191841', '680957', '157336']

path_curv = './../../Retinotopy/data/raw/converted'
# curv_old = scipy.io.loadmat(osp.join(path_curv, 'cifti_curv_all.mat'))[
#     'cifti_curv']

# For new curvature data:

# Defining number of nodes
number_cortical_nodes = int(64984)
number_hemi_nodes = int(number_cortical_nodes / 2)

# Loading new curvature data for the first subject in training set (background):
first_subject = 100610
new_data = nib.load(osp.join(path_curv, '../..', f'raw/converted/fs-curvature/{first_subject}/', \
    str(first_subject) + '.L.curvature.32k_fs_LR.shape.gii'))
new_data = torch.tensor(np.reshape(new_data.agg_data().reshape((number_hemi_nodes)), 
    (-1, 1)), dtype=torch.float)
# # Filter out NaNs if required
# new_data = new_data.masked_fill_(torch.tensor(np.reshape(torch.any(new_data.isnan(), dim=1).reshape((number_hemi_nodes)),
#     (-1, 1))), 0)
# # new_data = new_data[~torch.any(new_data.isnan())]
# Left hemisphere:
background = np.array(np.reshape(new_data.detach().numpy()[0:number_hemi_nodes], (-1)))


# Left hemisphere:
# background = np.zeros((number_hemi_nodes, 1))

# # Right hemisphere
# background = np.reshape(
#     curv['x' + hcp_id[subject_index] + '_curvature'][0][0][32492:], (-1))

threshold = 1  # threshold for the curvature map

# Background settings
nocurv = np.isnan(background)
background[nocurv == 1] = 0
background[background > 0] = 0.3
background[background < 0] = 0.5
# background[background < 0] = 0
# background[background > 0] = 1

# Loading training data
path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'Retinotopy/data')
pre_transform = T.Compose([T.FaceToEdge()])
# train_dataset_right = Retinotopy(path, 'Train', transform=T.Cartesian(),
#                                  pre_transform=pre_transform, n_examples=181,
#                                  prediction='polarAngle', myelination=True,
#                                  hemisphere='Right')
# train_loader_right = DataLoader(train_dataset_right, batch_size=1,
#                                 shuffle=True)

# For left hemisphere (no myelination):
train_dataset_left = Retinotopy(path, 'Train', transform=T.Cartesian(),
                                 pre_transform=pre_transform, n_examples=181,
                                 prediction='polarAngle', myelination=False,
                                 hemisphere='Left')
train_loader_left = DataLoader(train_dataset_left, batch_size=1,
                                shuffle=True)

# ROI settings
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
PolarAngle = np.zeros((32492, 1))

# Mean polar angle map
# PA = []
# for data in train_loader_right:
#     PA.append(np.array(data.y))
# PA = np.mean(PA, 0)

# For left hemisphere:
PA = []
for data in train_loader_left:
    PA.append(np.array(data.y))
PA = np.mean(PA, 0)

# Create an output folder if it doesn't already exist
directory = './output'
if not osp.exists(directory):
    osp.makedirs(directory)

# Saving the average map
np.savez('./output/AveragePolarAngleMap_RH.npz', list=PA)

# Settings for plot
# PolarAngle[final_mask_R == 1] = np.reshape(PA, (-1, 1))
# Left hemisphere:
PolarAngle[final_mask_L == 1] = np.reshape(PA, (-1, 1))
PolarAngle = np.array(PolarAngle)
minus = PolarAngle > 180
sum = PolarAngle < 180
PolarAngle[minus] = PolarAngle[minus] - 180
PolarAngle[sum] = PolarAngle[sum] + 180

# Masking
# PolarAngle[final_mask_R == 1] += threshold
# PolarAngle[final_mask_R != 1] = 0
# Left hemisphere:
PolarAngle[final_mask_L == 1] += threshold
PolarAngle[final_mask_L != 1] = 0

view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(PolarAngle[0:32492], (-1)), bg_map=background,
    cmap='gist_rainbow', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=361,
    title=f'Polar angle Left hemisphere mean ground truth (training set)')
view.open_in_browser()
view.save_as_html(f'D:\\Retinotopy Project\\surf_images\\PA_LH_mean_trainset\\empirical_mean_trainset_LH')



# # Left hemisphere
# background = np.reshape(
#     curv['x' + hcp_id[subject_index] + '_curvature'][0][0][0:32492], (-1))

# threshold = 1  # threshold for the curvature map

# # Background settings
# nocurv = np.isnan(background)
# background[nocurv == 1] = 0
# background[background < 0] = 0
# background[background > 0] = 1

# # Loading training data
# train_dataset_left = Retinotopy(path, 'Train', transform=T.Cartesian(),
#                                 pre_transform=pre_transform, n_examples=181,
#                                 prediction='polarAngle', myelination=True,
#                                 hemisphere='Left')
# train_loader_left = DataLoader(train_dataset_left, batch_size=1, shuffle=True)

# # ROI settings
# label_primary_visual_areas = ['ROI']
# final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
#     label_primary_visual_areas)
# PolarAngle = np.zeros((32492, 1))

# # Mean polar angle map
# PA = []
# for data in train_loader_left:
#     PA.append(np.array(data.y))
# PA = np.mean(PA, 0)

# # Saving the average map
# np.savez('./output/AveragePolarAngleMap_LH.npz', list=PA)

# # Settings for plot
# PolarAngle[final_mask_L == 1] = np.reshape(PA, (-1, 1))
# PolarAngle = np.array(PolarAngle)
# minus = PolarAngle > 180
# sum = PolarAngle < 180
# PolarAngle[minus] = PolarAngle[minus] - 180
# PolarAngle[sum] = PolarAngle[sum] + 180

# # Masking
# PolarAngle[final_mask_L == 1] += threshold
# PolarAngle[final_mask_L != 1] = 0

# view = plotting.view_surf(
#     surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
#                        'Retinotopy/data/raw/surfaces'
#                        '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
#     surf_map=np.reshape(PolarAngle[0:32492], (-1)), bg_map=background,
#     cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
#     threshold=threshold, vmax=361)
# view.open_in_browser()