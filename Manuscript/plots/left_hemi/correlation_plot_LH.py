import os
import os.path as osp
import torch
import torch_geometric.transforms as T
import sys
import scipy.io
import numpy as np
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


"""
The code in this file was used to calculate and graph the Pearson correlation
coefficient for Left hemisphere curvature data of each participant in the HCP 
dataset. This specifically measures the correlation between curvature data 
processed using the HCP-specific processing pipeline, and curvature data 
processed using a more typical, 'standard' processing pipeline. The visual 
mask has been removed to calculate the correlations between curvature data.

Note: code implementation assumes that the file is being run from the dir 
Manuscript/plots/left_hemi - I have modified the code to automatically set the 
working dir to this (if it isn't already).
"""
# Set the working directory to Manuscript/plots/left_hemi
os.chdir(osp.dirname(osp.realpath(__file__)))

# Total number of cortical nodes in the mesh
NUMBER_CORTICAL_NODES = int(64984)
# Number of nodes within each hemisphere
NUMBER_HEMI_NODES = int(NUMBER_CORTICAL_NODES / 2)

# Configure filepaths
sys.path.append('..')
# Location of participants' curvature data
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', '..', 
                    'Retinotopy', 'data', 'raw', 'converted')


# Loading HCP participant IDs
with open(osp.join(path, '..', '..', 'list_subj')) as fp:
    subj = fp.read().split("\n")
subj = subj[0:len(subj) - 1]


# Loading curvature data with the HCP-specific processing pipeline applied
curv_hcp = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']

curv_hcp_values = []
curv_std_values = []

for index in range(0, len(subj)):
    # Reading HCP processing pipeline curvature data with visual mask removed
    data_hcp = torch.tensor(np.reshape(
        curv_hcp['x' + subj[index] + '_curvature'][0][0][
        0:NUMBER_HEMI_NODES].reshape(
            (NUMBER_HEMI_NODES)), (-1, 1)),
        dtype=torch.float)
    # Filter out NaNs
    data_hcp = data_hcp.masked_fill_(torch.tensor(np.reshape(torch.any(
        data_hcp.isnan(), dim=1).reshape((NUMBER_HEMI_NODES)), (-1, 1))), 0)
    curv_hcp_values.append(data_hcp)

    # Reading standard processing pipeline curvature data (visual mask removed)
    data_std = nib.load(osp.abspath(osp.join(path, 'fs-curvature',
        f'{subj[index]}', subj[index] + '.L.curvature.32k_fs_LR.shape.gii')))
    data_std = torch.tensor(np.reshape(data_std.agg_data()
        .reshape((NUMBER_HEMI_NODES)), (-1, 1)), dtype=torch.float)
    # Filter out NaNs
    data_std = data_std.masked_fill_(torch.tensor(np.reshape(torch.any(
        data_std.isnan(), dim=1).reshape((NUMBER_HEMI_NODES)),(-1, 1))), 0)
    curv_std_values.append(data_std)

# Calculate the Pearson correlation coefficient
correlations = np.asarray([np.corrcoef(curv_std_values[i].T, 
    curv_hcp_values[i].T[0].T, dtype=float)[0][1] 
    for i in range (0, len(subj))])


#### Plotting data: ####

# Create a Pandas DataFrame for the graph
df = pd.DataFrame({'Subject': [i for i in range(0, len(subj))], 
        'Correlation': correlations})

# Create an output folder if it doesn't already exist
directory = './output'
if not os.path.exists(directory):
    os.makedirs(directory)

# Save correlation data to text file
# f = open('./output/correlations_LH.txt', "w")
# f.write('Correlation data (Left hemisphere)\n')
# f.writelines([f'Subject: {i}; ID: {subj[i]}; \
#   Correlation: {correlations[i]}\n' for i in range(0, len(subj))])
# f.close()

# Graph settings
sns.set(style="darkgrid")
sns
sns.regplot(df, x=df['Subject'], y=df['Correlation'], fit_reg=True)
# Save to file
# plt.savefig('./output/correlations_LH.png', dpi=300)

# Show graph
plt.show()


