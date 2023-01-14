import os.path as osp
import scipy.io
import sys
import torch_geometric.transforms as T
import numpy as np

# For reading new curvature data
import nibabel as nib
import torch

sys.path.append('../..')

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from nilearn import plotting
from Retinotopy.dataset.HCP_3sets_ROI import Retinotopy
from torch_geometric.data import DataLoader

# cd Manuscript/plots
path = './../../Retinotopy/data/raw/converted'

# cd Manuscript/plots/left_hemi
# path = './../../../Retinotopy/data/raw/converted'
curv_old_raw = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
background = np.reshape(curv_old_raw['x100610_curvature'][0][0][0:32492], (-1))

# For new curvature data:
# Loading subject IDs
with open(osp.join(path, '../..', 'list_subj')) as fp:
    subjects = fp.read().split("\n")
subjects = subjects[0:len(subjects) - 1]
# Defining number of nodes
number_cortical_nodes = int(64984)
number_hemi_nodes = int(number_cortical_nodes / 2)

# Reading all data
curv_old = []
curv_new = []
for index in range(0, len(subjects)):
    # Reading old curvature data - visual mask removed (Left hemi)
    old_data = torch.tensor(np.reshape(
        curv_old_raw['x' + subjects[index] + '_curvature'][0][0][
        0:number_hemi_nodes].reshape(
            (number_hemi_nodes)), (-1, 1)),
        dtype=torch.float)
    # # Filter out NaNs
    # old_data = old_data.masked_fill_(torch.tensor(np.reshape(torch.any(old_data.isnan(), dim=1).reshape((number_hemi_nodes)),
    #     (-1, 1))), 0)
    # old_data = old_data[~torch.any(old_data.isnan())]
    curv_old.append(old_data)

    # Reading new curvature data (Left hemi)
    new_data = nib.load(osp.join(path, '../..', f'raw/converted/fs-curvature/{subjects[index]}/', \
        subjects[index] + '.L.curvature.32k_fs_LR.shape.gii'))
    new_data = torch.tensor(np.reshape(new_data.agg_data().reshape((number_hemi_nodes)), 
        (-1, 1)), dtype=torch.float)
    # # Filter out NaNs
    # new_data = new_data.masked_fill_(torch.tensor(np.reshape(torch.any(new_data.isnan(), dim=1).reshape((number_hemi_nodes)),
    #     (-1, 1))), 0)
    # # new_data = new_data[~torch.any(new_data.isnan())]
    curv_new.append(new_data)
# background_new = np.zeros((number_hemi_nodes, 1))
background_new = np.array(np.reshape(curv_new[0][0:32492], (-1)))

# Background settings
threshold = 1
# threshold = 0
nocurv = np.isnan(background)
background[nocurv == 1] = 0

# New curvature background settings
nocurv_new = np.isnan(background_new)
background_new[nocurv_new == 1] = 0

# ROI settings
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
# print(np.sum(final_mask_L==1))
R2_thr = np.zeros((32492, 1))
# For new curvature
R2_thr_new = np.zeros((32492, 1))

# Loading data - left hemisphere
# path = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'Retinotopy/data')
# pre_transform = T.Compose([T.FaceToEdge()])
# test_dataset = Retinotopy(path, 'Test', transform=T.Cartesian(),
#                            pre_transform=pre_transform, n_examples=181,
#                            prediction='polarAngle', myelination=True,
#                            hemisphere='Left')
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Loading old data (left hemisphere) without visual mask applied
test_loader = DataLoader(curv_old, batch_size=1, shuffle=True)

# Creating dataset for new curvature = left hemisphere
test_loader_new = DataLoader(curv_new, batch_size=1, shuffle=True)

# # Average explained variance map
# R2 = []
# for data in test_loader:
#     R2.append(np.array(data.R2))
# R2 = np.mean(R2, 0)

# # New curvature - Average explained variance map
# R2_new = []
# for new_data in test_loader_new:
#     R2_new.append(np.array(new_data.R2))
# R2_new = np.mean(R2_new, 0)

# # Masking
# R2_thr[final_mask_L == 1] = np.reshape(R2, (-1, 1)) + threshold
# R2_thr[final_mask_L != 1] = 0

# Masking
R2_thr[final_mask_L == 1] = np.divide(np.power(5, np.reshape(np.asarray(curv_old[0][final_mask_L == 1]), (-1, 1)) + 10), 100000)
R2_thr[final_mask_L != 1] = 0

# New curvature - Masking
R2_thr_new[final_mask_L == 1] = np.divide(np.power(5, np.reshape(np.asarray(curv_new[0][final_mask_L == 1]), (-1, 1))+ 10), 100000)
R2_thr_new[final_mask_L != 1] = 0

'''
Why all of the random operations on the curvature values?
- Adding 10 to the curvature values to ensure that they are non-negative. This prevents surf_map from
changing the symmetric_cmap setting variable from False to true. The map looks funny and inaccurate
if it is set to symmetric.
- Converting every individual curvature value to 5^curvval. Basically trying to 'expand' the domain
of curvature values so that there is a greater variance between the lowest values and highest values.
This ensures that the value variations show up more distinctly on surf_map.
- Dividing curvature values by 100,000 - more of an arbitrary thing that doesn't affect the visualisation.
Only done to ensure that the range of values in the colourbar appears reasonable (not in the millions)
The end goal of these operations is to 'expand' the values such that their coloured visualisation is
easier to distinguish.
'''


# view = plotting.view_surf(
#     surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
#                        'Retinotopy/data/raw/surfaces'
#                        '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
#     bg_map=background, surf_map=np.reshape(R2_thr[0:32492], (-1)),
#     threshold=threshold, cmap='hot', black_bg=False, symmetric_cmap=False,
#     vmax=60 + threshold)
# view.open_in_browser()

# print(min([R2_thr[i][0] for i in range(0, len(R2_thr)) if R2_thr[i][0] > 0]))
# print(min([R2_thr_new[i][0] for i in range(0, len(R2_thr_new)) if R2_thr_new[i][0] > 0]))
# print(np.mean(R2_thr))
# print(np.mean(R2_thr_new))
# correlation = np.corrcoef(R2_thr[final_mask_L == 1].flatten(), R2_thr_new[final_mask_L == 1].flatten())
# print(correlation)

# surf_map should be loading the curvature!
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    bg_map=background, surf_map=np.reshape(R2_thr[0:32492], (-1)), 
    threshold=threshold, cmap='plasma', black_bg=False, symmetric_cmap=False, title='sdfds')
view.open_in_browser() #vmin=190

# For new curvature - view
view_new = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    bg_map=background_new, surf_map=np.reshape(R2_thr_new[0:32492], (-1)),
    threshold=threshold, cmap='plasma', black_bg=False, symmetric_cmap=False, title='sf')
view_new.open_in_browser()