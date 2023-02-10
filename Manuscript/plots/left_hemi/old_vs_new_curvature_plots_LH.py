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


# Will scaling be applied to the image?
scaling = False

'''
If subject_to_plot is set to None, plots are generated for all subjects.
Otherwise, one plot is generated for the subject at the given int index.
'''
subject_to_plot = 0

# Defining number of nodes
number_cortical_nodes = int(64984)
number_hemi_nodes = int(number_cortical_nodes / 2)

# cd Manuscript/plots
path = './../../Retinotopy/data/raw/converted'

curv_old_raw = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']

# For new curvature data:
# Loading subject IDs
with open(osp.join(path, '../..', 'list_subj')) as fp:
    subjects = fp.read().split("\n")
subjects = subjects[0:len(subjects) - 1]

# Reading all data
curv_old = []
curv_new = []
for index in range(0, len(subjects)):
    # Reading old curvature data - visual mask removed (Left hemi)
    old_data = torch.tensor(np.reshape(
        curv_old_raw['x' + subjects[index] + '_curvature'][0][0][
        0:number_hemi_nodes].reshape((number_hemi_nodes)), (-1, 1)),
        dtype=torch.float)
    curv_old.append(old_data)

    # Reading new curvature data (Left hemi)
    new_data = nib.load(osp.join(path, f'fs-curvature/{subjects[index]}/', 
        f'{subjects[index]}.L.curvature.32k_fs_LR.shape.gii'))
    new_data = torch.tensor(np.reshape(new_data.agg_data()
        .reshape((number_hemi_nodes)), (-1, 1)), dtype=torch.float)
    curv_new.append(new_data)

def plot_curvatures(subject_index, scaling):
    """
    Generates curvature surf plots for the subject, with the respective
    background mesh applied. The surf plot can be viewed in a web browser
    and/or saved as a HTML file.
    Args:
        subject_index (int): Index of participant to generate plot for
        scaling (boolean): Will a non-linear scaling be applied to
                            the curvature values? Used to generate a more
                            distinct/contrasting colour scheme for the plot.
    """
    background_new = np.array(np.reshape(curv_new[subject_index][
        0:number_hemi_nodes], (-1)))
    background = np.reshape(curv_old_raw['x' + subjects[subject_index] + 
        '_curvature'][0][0][0:number_hemi_nodes], (-1))

    # Background settings
    threshold = 1
    nocurv = np.isnan(background)
    background[nocurv == 1] = 0

    # New curvature background settings
    nocurv_new = np.isnan(background_new)
    background_new[nocurv_new == 1] = 0

    # ROI settings
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
        
    curv_old_masked = np.zeros((number_hemi_nodes, 1))
    curv_new_masked = np.zeros((number_hemi_nodes, 1))

    # Creating loader for old curvature - left hemisphere
    old_loader = DataLoader(curv_old, batch_size=1, shuffle=True)

    # Creating loader for new curvature - left hemisphere
    new_loader = DataLoader(curv_new, batch_size=1, shuffle=True)

    if scaling:
        '''
        Why all of the random scaling operations on the curvature values?
        - Adding 10 to the curvature values to ensure that they are 
        non-negative. This prevents surf_map from changing the symmetric_cmap 
        setting variable from False to true. The map looks funny and inaccurate
        if it is set to symmetric.
        - Converting every individual curvature value to 5^curvval. 
        Basically trying to 'expand' the domain of curvature values so that 
        there is a greater variance between the lowest values and highest 
        values. This ensures that the value variations show up more distinctly 
        on surf_map.
        - Dividing curvature values by 100,000 - more of an arbitrary thing 
        that doesn't affect the visualisation. Only done to ensure that the 
        range of values in the colourbar appears reasonable (not in the millions)
        The end goal of these operations is to 'expand' the values such that 
        their coloured visualisation is easier to distinguish.
        '''
        # Old curvature - Masking w/ scaling applied
        curv_old_masked[final_mask_L == 1] = np.divide(np.power(5, 
            np.reshape(np.asarray(curv_old[subject_index][final_mask_L == 1]), 
            (-1, 1)) + 10), 100000)
        curv_old_masked[final_mask_L != 1] = 0

        # New curvature - Masking w/ scaling applied
        curv_new_masked[final_mask_L == 1] = np.divide(np.power(5, 
            np.reshape(np.asarray(curv_new[subject_index][final_mask_L == 1]), 
            (-1, 1))+ 10), 100000)
        curv_new_masked[final_mask_L != 1] = 0

        scaling_dir = 'rescaled'
    else:
        # No re-scaling applied
        # Masking - only translating values up
        curv_old_masked[final_mask_L == 1] = (np.reshape(np.asarray(
            curv_old[subject_index][final_mask_L == 1]), (-1, 1)) + 1.5)
        curv_old_masked[final_mask_L != 1] = 0

        # New curvature - Masking - only translating values up
        curv_new_masked[final_mask_L == 1] = (np.reshape(np.asarray(
            curv_new[subject_index][final_mask_L == 1]), (-1, 1)) + 1.5)
        curv_new_masked[final_mask_L != 1] = 0

        scaling_dir = 'no_scaling'

    # For old curvature
    view = plotting.view_surf(
        surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                    'Retinotopy/data/raw/surfaces'
                    f'/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
        bg_map=background, surf_map=np.reshape(curv_old_masked[
        0:number_hemi_nodes], (-1)), threshold=threshold, cmap='plasma', 
        black_bg=False, symmetric_cmap=False, title=f'Subject \
        {subjects[subject_index]}: Old curvature data (Left hemisphere)')
    # View in browser
    view.open_in_browser()
    # Save as html file
    # view.save_as_html(f'D:\\Retinotopy Project\\surf_images\\\
    #     old_vs_new_curvature\\{scaling_dir}\\\
    #     {subjects[subject_index]}_LH_oldcurv')

    # For new curvature
    view_new = plotting.view_surf(
        surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                        'Retinotopy/data/raw/surfaces'
                        f'/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
        bg_map=background_new, surf_map=np.reshape(curv_new_masked[
        0:number_hemi_nodes], (-1)), threshold=threshold, cmap='plasma', 
        black_bg=False, symmetric_cmap=False, title=f'Subject \
        {subjects[subject_index]}: New curvature data (Left hemisphere)')
    # View in browser
    view_new.open_in_browser()
    # Save as html file
    # view_new.save_as_html(f'D:\\Retinotopy Project\\surf_images\\\
    #     old_vs_new_curvature\\{scaling_dir}\\\
    #     {subjects[subject_index]}_LH_newcurv')


if subject_to_plot is None:
    # Plotting curvatures of all subjects
    for subject_index in range(0, len(subjects)):
        plot_curvatures(subject_index=subject_index, scaling=scaling)
else:
    # Plotting curvature of a single subject
    plot_curvatures(subject_index=subject_to_plot, scaling=scaling)