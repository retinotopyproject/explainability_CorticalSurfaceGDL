import os.path as osp
import scipy.io
import sys
import torch_geometric.transforms as T
import numpy as np

sys.path.append('../..')

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from nilearn import plotting
from Retinotopy.dataset.HCP_stdprocessing_3sets_ROI import Retinotopy
from torch_geometric.data import DataLoader

# For reading new curvature data
import nibabel as nib
import torch

path = './../../../Retinotopy/data/raw/converted'
# For loading old curvature data for first subject (background):
# curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
# background = np.reshape(curv['x100610_curvature'][0][0][0:32492], (-1))

# Defining number of nodes
number_cortical_nodes = int(64984)
number_hemi_nodes = int(number_cortical_nodes / 2)

# Loading new curvature data for first subject (background):
first_subject = 100610
new_data = nib.load(osp.join(path, '../..', f'raw/converted/fs-curvature/{first_subject}/', \
    str(first_subject) + '.R.curvature.32k_fs_LR.shape.gii'))
new_data = torch.tensor(np.reshape(new_data.agg_data().reshape((number_hemi_nodes)), 
    (-1, 1)), dtype=torch.float)
# # Filter out NaNs if required
# new_data = new_data.masked_fill_(torch.tensor(np.reshape(torch.any(new_data.isnan(), dim=1).reshape((number_hemi_nodes)),
#     (-1, 1))), 0)
# # new_data = new_data[~torch.any(new_data.isnan())]
background = np.reshape(new_data, (-1)).detach().numpy()

# Background settings
threshold = 1
nocurv = np.isnan(background)
background[nocurv == 1] = 0

# ROI settings
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
R2_thr = np.zeros((32492, 1))

# Loading data - left hemisphere
path = osp.join(osp.dirname(osp.realpath(__file__)), '../../..', 'Retinotopy/data')
pre_transform = T.Compose([T.FaceToEdge()])
# test_dataset = Retinotopy(path, 'Test', transform=T.Cartesian(),
#                            pre_transform=pre_transform, n_examples=181,
#                            prediction='polarAngle', myelination=True,
#                            hemisphere='Right')
# For curvature data only (myelination=False)
test_dataset = Retinotopy(path, 'Test', transform=T.Cartesian(),
                           pre_transform=pre_transform, n_examples=181,
                           prediction='polarAngle', myelination=False,
                           hemisphere='Right')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Average explained variance map
R2 = []
for data in test_loader:
    R2.append(np.array(data.R2))
R2 = np.mean(R2, 0)

# Masking
R2_thr[final_mask_R == 1] = np.reshape(R2, (-1, 1)) + threshold
R2_thr[final_mask_R != 1] = 0

view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.R.sphere.32k_fs_LR.surf.gii'),
    bg_map=background, surf_map=np.reshape(R2_thr[0:32492], (-1)),
    threshold=threshold, cmap='hot', black_bg=False, symmetric_cmap=False,
    vmax=60 + threshold)
view.open_in_browser()
view.save_as_html(f'D:\\Retinotopy Project\\surf_images\\R2_average\\R2_average_RH')