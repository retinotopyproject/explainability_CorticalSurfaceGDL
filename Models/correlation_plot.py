import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import sys
import time

import scipy.io
import numpy as np
import nibabel as nib

sys.path.append('..')

from Retinotopy.dataset.HCP_3sets_ROI import Retinotopy
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv

'''
Setting up the training dataset and previously used datasets,
checking the correlation between them.
'''

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Retinotopy', 'data')
# path = '/home/user/Desktop/storage/containers/deepretinotopy_1.0.0_20221201/explainability_CorticalSurfaceGDL/Retinotopy/data/'
norm_value = 70.4237
pre_transform = T.Compose([T.FaceToEdge()])

train_dataset = Retinotopy(path, 'Train', transform=T.Cartesian(max_value=norm_value),
                            pre_transform=pre_transform, n_examples=181,
                           prediction='polarAngle', myelination=False,
                           hemisphere='Left') # Change to Right for the RH
                           
prev_dataset = Retinotopy(path, 'Train', transform=T.Cartesian(max_value=norm_value),
                            pre_transform=pre_transform, n_examples=181,
                           prediction='polarAngle', myelination=True,
                           hemisphere='Left') # Change to Right for the RH

# Loading subject IDs
with open(osp.join(path, 'list_subj')) as fp:
    subjects = fp.read().split("\n")
subjects = subjects[0:len(subjects) - 1]


#### copied from read_HCPdata ####

# Loading the measures
curv = scipy.io.loadmat(osp.join(path, 'raw', 'converted', 'cifti_curv_all.mat'))['cifti_curv']

# Defining number of nodes
number_cortical_nodes = int(64984)
number_hemi_nodes = int(number_cortical_nodes / 2)

prev_curvature = []
new_curvature = []

for index in range(0, len(subjects)):
    # Reading old curvature data - visual mask removed (Right hemi)
    old_data = torch.tensor(np.reshape(
        curv['x' + subjects[index] + '_curvature'][0][0][
        number_hemi_nodes:number_cortical_nodes].reshape(
            (number_hemi_nodes)), (-1, 1)),
        dtype=torch.float)
    # Filter out NaNs
    old_data = old_data.masked_fill_(torch.tensor(np.reshape(torch.any(old_data.isnan(), dim=1).reshape((number_hemi_nodes)),
        (-1, 1))), 0)
    # old_data = old_data[~torch.any(old_data.isnan())]
    prev_curvature.append(old_data)

    # Reading new curvature data - removing visual mask (Right hemi)
    new_data = nib.load(osp.join(path, f'raw/converted/fs-curvature/{subjects[index]}/', \
        subjects[index] + '.R.curvature.32k_fs_LR.shape.gii'))
    new_data = torch.tensor(np.reshape(new_data.agg_data().reshape((number_hemi_nodes)), 
        (-1, 1)), dtype=torch.float)
    # Filter out NaNs
    new_data = new_data.masked_fill_(torch.tensor(np.reshape(torch.any(new_data.isnan(), dim=1).reshape((number_hemi_nodes)),
        (-1, 1))), 0)
    # new_data = new_data[~torch.any(new_data.isnan())]
    new_curvature.append(new_data)


# train_curv = [train_dataset[i].x for i in range(0, len(train_dataset))]
# prev_curv = [torch.transpose(prev_dataset[i].x, 0, 1)[0] for i in range(0, len(prev_dataset))]
# correlations = np.asarray([np.corrcoef(train_dataset[i].x.T,prev_dataset[i].x.T[0].T)[0][1] for i in range (0, len(train_dataset))])

# print(train_dataset[0].x.T.size())
# print(new_curvature[0].T.size())
# print(prev_dataset[0].x.T[0].T.size())
# print(prev_curvature[0].T[0].T.size())
# print(np.corrcoef(new_curvature[0].T, prev_curvature[0].T[0].T, dtype=float))
# print(np.corrcoef(train_dataset[0].x.T,prev_dataset[0].x.T[0].T, dtype=float))
correlations = np.asarray([np.corrcoef(new_curvature[i].T, prev_curvature[i].T[0].T, dtype=float)[0][1] for i in range (0, len(subjects))])
print(correlations)

'''
Plotting data
'''
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create dataframe
# df = pd.DataFrame({'Subject': [i for i in range(0, len(train_dataset))], 'Correlation': correlations})
df = pd.DataFrame({'Subject': [i for i in range(0, len(subjects))], 'Correlation': correlations})

# Set a grey background
sns.set(style="darkgrid")
sns
sns.regplot(df, x=df['Subject'], y=df['Correlation'], fit_reg=True)
plt.savefig('correlations.png', dpi=300)


low_correlation = {f'Subject ID: {subjects[i]} Subject index in list: {i}': correlations[i] for i in range(0, len(correlations)) if correlations[i] <= 0.4}
print(low_correlation)
plt.show()


