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

norm_value = 70.4237

# Defining number of nodes
number_cortical_nodes = int(64984)
number_hemi_nodes = int(number_cortical_nodes / 2)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Retinotopy', 'data')
pre_transform = T.Compose([T.FaceToEdge()])

'''
Setting up the training dataset and previously used dataset (that contains
myelination data), checking the correlation between them.
'''
# train_dataset = Retinotopy(path, 'Train', transform=T.Cartesian(max_value=norm_value),
#                             pre_transform=pre_transform, n_examples=181,
#                            prediction='polarAngle', myelination=False,
#                            hemisphere='Right') # Change to Left for the LH
                           
# prev_dataset = Retinotopy(path, 'Train', transform=T.Cartesian(max_value=norm_value),
#                             pre_transform=pre_transform, n_examples=181,
#                            prediction='polarAngle', myelination=True,
#                            hemisphere='Right') # Change to Left for the LH

# train_curv = [train_dataset[i].x for i in range(0, len(train_dataset))]
# prev_curv = [torch.transpose(prev_dataset[i].x, 0, 1)[0] 
#               for i in range(0, len(prev_dataset))]
# correlations = np.asarray([np.corrcoef(train_dataset[i].x.T, 
#       prev_dataset[i].x.T[0].T)[0][1] for i in range (0, len(train_dataset))])

# Loading subject IDs
with open(osp.join(path, 'list_subj')) as fp:
    subjects = fp.read().split("\n")
subjects = subjects[0:len(subjects) - 1]

# For checking correlations with unprocessed data (no visual masks applied):

# Loading the measures
curv = scipy.io.loadmat(osp.join(path, 'raw', 'converted', 
        'cifti_curv_all.mat'))['cifti_curv']

old_curvature = []
new_curvature = []

for index in range(0, len(subjects)):
    # Reading old curvature data - visual mask removed (Right hemi)
    old_data = torch.tensor(np.reshape(
        curv['x' + subjects[index] + '_curvature'][0][0][
        number_hemi_nodes:number_cortical_nodes].reshape(
            (number_hemi_nodes)), (-1, 1)),
        dtype=torch.float)
    # Filter out NaNs
    old_data = old_data.masked_fill_(torch.tensor(np.reshape(torch.any(
        old_data.isnan(), dim=1).reshape((number_hemi_nodes)), (-1, 1))), 0)
    old_curvature.append(old_data)

    # Reading new curvature data with no visual mask applied (Right hemi)
    new_data = nib.load(osp.join(path, f'raw/converted/fs-curvature/\
        {subjects[index]}/', subjects[index] + 
        '.R.curvature.32k_fs_LR.shape.gii'))
    new_data = torch.tensor(np.reshape(new_data.agg_data()
        .reshape((number_hemi_nodes)), (-1, 1)), dtype=torch.float)
    # Filter out NaNs
    new_data = new_data.masked_fill_(torch.tensor(np.reshape(torch.any(
        new_data.isnan(), dim=1).reshape((number_hemi_nodes)), (-1, 1))), 0)
    new_curvature.append(new_data)

correlations = np.asarray([np.corrcoef(new_curvature[i].T, 
    old_curvature[i].T[0].T, dtype=float)[0][1] 
    for i in range (0, len(subjects))])

'''
Plotting data
'''
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create dataframe
df = pd.DataFrame({'Subject': [i for i in range(0, len(subjects))], 
    'Correlation': correlations})

# Save correlation data to text file
# f = open('correlations_RH.txt', "w")
# f.write('Correlation data (Right hemisphere)\n')
# f.writelines([f'Subject: {i}; ID: {subjects[i]}; \
#     Correlation: {correlations[i]}\n' for i in range(0, len(subjects))])
# f.close()

# Graph settings
sns.set(style="darkgrid")
sns
sns.regplot(df, x=df['Subject'], y=df['Correlation'], fit_reg=True)
# Save to file
plt.savefig('correlations_RH.png', dpi=300)

# Show subjects with correlation values <=0.4
# low_correlation = {f'Subject ID: {subjects[i]} Subject index in list: {i}': 
#     correlations[i] for i in range(0, len(correlations)) 
#     if correlations[i] <= 0.4}
# print(low_correlation)

# Show graph
plt.show()
