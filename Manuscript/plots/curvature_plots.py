import os.path as osp
import os
import scipy.io
import sys
import torch
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import numpy as np
import nibabel as nib
from nilearn import plotting

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from Retinotopy.dataset.HCP_stdprocessing_3sets_ROI import Retinotopy


"""
This file was used to graph 
TODO

Note: code implementation assumes that the file is being run from the dir 
explainability_CorticalSurfaceGDL/Manuscript/plots - I have modified 
the code to automatically set the working dir to this (if it isn't already).
"""
# Set the working directory to Manuscript/plots
os.chdir(osp.join(osp.dirname(osp.realpath(__file__))))


# Total number of cortical nodes in the mesh
NUMBER_CORTICAL_NODES = int(64984)
# Number of nodes within each hemisphere
NUMBER_HEMI_NODES = int(NUMBER_CORTICAL_NODES / 2)

# Configure filepaths
sys.path.append('../..')


def read_HCP_curv(hemisphere='Left'):
    """
    Loads the curvature data for every participant in the HCP dataset,
    for the specified hemisphere.
    The curvature data to be loaded has been processed with the HCP-specific
    pre-processsing pipeline.

    Args:
        hemisphere (str): Which hemisphere will be plotted? 'Left' or 'Right'

    Outputs:
        subj: the list of participant IDs in the HCP dataset
        curv: the curvature data for every HCP participant (HCP-specific
              processing pipeline)
        curv_raw: the raw curvature data, extracted from the Matlab file
                  'cifti_curv_all.mat'
    """
    # Path to HCP participants' curvature data
    HCP_path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 
                    'Retinotopy/data/raw/converted')

    # Load the raw curvature data
    curv_raw = scipy.io.loadmat(osp.join(HCP_path, 
                                        'cifti_curv_all.mat'))['cifti_curv']

    # Loading participant IDs:
    with open(osp.join(HCP_path, '../..', 'list_subj')) as fp:
        subj = fp.read().split("\n")
    subj = subj[0:len(subj) - 1]

    # Read the curvature data for each participant
    curv = []
    for index in range(0, len(subj)):
        '''
        Note: the Left hemisphere is made up of nodes with indices from the first
        node (at index 0) up to NUMBER_HEMI_NODES - 1. 
        The Right hemisphere contains nodes with indices from NUMBER_HEMI_NODES 
        up to NUMBER_CORTICAL_NODES - 1.
        '''
        # Read the participant data with the visual mask removed
        if hemisphere == 'Left':
            data = torch.tensor(np.reshape(
                curv_raw['x' + subj[index] + '_curvature'][0][0][
                0:NUMBER_HEMI_NODES].reshape((NUMBER_HEMI_NODES)), (-1, 1)),
                dtype=torch.float)
        else:
            # hemisphere == 'Right'
            data = torch.tensor(np.reshape(
                curv_raw['x' + subj[index] + '_curvature'][0][0][
                NUMBER_HEMI_NODES:].reshape((NUMBER_HEMI_NODES)), (-1, 1)),
                dtype=torch.float)
        # Add the participant's data to the curvature data list
        curv.append(data)

    return subj, curv, curv_raw


def plot_HCP_curv(subj_index=0, hemisphere='Left', view_plot=True):
    """
    Plot the curvature map for a given participant from the HCP dataset.
    This method plots curvature values which have been processed with the
    HCP-specific pre-processing pipeline.
    Args:
        subject_index (int): Index of participant to generate plot for.
                             eg. if subject_index == 0, plot the curvature map
                             for the first participant.
        hemisphere (str): Which hemisphere will be plotted? 'Left' or 'Right'
        view_plot (bool): If true, show the curvature plot as a HTML file
                          in a web browser. If false, don't show the plot
    """
    # Get the list of participants, curvature data (and raw curv data)
    subj, curv, curv_raw = read_HCP_curv(subj_index)

    '''
    Note: the Left hemisphere is made up of nodes with indices from the first
    node (at index 0) up to NUMBER_HEMI_NODES - 1. 
    The Right hemisphere contains nodes with indices from NUMBER_HEMI_NODES 
    up to NUMBER_CORTICAL_NODES - 1.
    '''
    # Create the background curvature map
    if hemisphere == 'Left':
        background = np.reshape(curv_raw['x' + subj[subj_index] + 
                                '_curvature'][0][0][0:NUMBER_HEMI_NODES], (-1))
    else:
        # hemisphere == 'Right'
        background = np.reshape(curv_raw['x' + subj[subj_index] + 
                                '_curvature'][0][0][NUMBER_HEMI_NODES:], (-1))
    
    threshold = 1 # Threshold for the curvature map

    # Remove NaNs from curvature map
    nocurv = np.isnan(background)
    background[nocurv == 1] = 0
    # Background settings (discretize curvature values to give a 2 colour map)
    background[background < 0] = 0
    background[background > 0] = 1

    # Selecting all visual areas (Wang2015) plus V1-3 fovea
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
    # Apply ROI mask for the relevant hemisphere
    if hemisphere == 'Left':
        final_mask = final_mask_L
    else:
        # hemisphere == 'Right'
        final_mask = final_mask_R

    # Applying masking
    curv_masked = np.zeros((NUMBER_HEMI_NODES, 1))
    curv_masked[final_mask == 1] = (np.reshape(np.asarray(
                                        curv[subj_index][final_mask == 1]), 
                                        (-1, 1)) + 1.5)
    curv_masked[final_mask != 1] = 0


    #### Plotting the curvature data ####
    view = plotting.view_surf(
        surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                    'Retinotopy/data/raw/surfaces',
                    f'S1200_7T_Retinotopy181.{hemisphere[0]}' +
                    '.sphere.32k_fs_LR.surf.gii'),
        bg_map=background, surf_map=np.reshape(curv_masked[
        0:NUMBER_HEMI_NODES], (-1)), threshold=threshold, cmap='plasma', 
        black_bg=False, symmetric_cmap=False,
        title=f'HCP participant \
        {subj[subj_index]}: curvature data (HCP-processing pipeline, \
        {hemisphere} hemisphere)')
    
    if view_plot:
        # View plot in web browser
        view.open_in_browser()


def read_HCP_stdprocessing_curv(hemisphere='Left'):
    """
    Loads the curvature data for every participant in the HCP dataset,
    for the specified hemisphere.
    The curvature data to be loaded has been processed with a more standardised
    pre-processsing pipeline than the HCP-specific pipeline.

    Args:
        hemisphere (str): Which hemisphere will be plotted? 'Left' or 'Right'

    Outputs:
        subj: the list of participant IDs in the HCP dataset
        curv: the curvature data for every HCP participant (standard
              processing pipeline)
    
    """
    # Path to HCP participants' curvature data
    HCP_path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 
                    'Retinotopy/data/raw/converted')

    # Loading participant IDs:
    with open(osp.join(HCP_path, '../..', 'list_subj')) as fp:
        subj = fp.read().split("\n")
    subj = subj[0:len(subj) - 1]

    # Read the curvature data for each participant
    curv = []
    for index in range(0, len(subj)):
        # Read the participants' data for the relevant hemisphere
        data = nib.load(osp.join(HCP_path, f'fs-curvature/{subj[index]}/', \
            f'{subj[index]}.{hemisphere[0]}.curvature.32k_fs_LR.shape.gii'))
        data = torch.tensor(np.reshape(data.agg_data()
            .reshape((NUMBER_HEMI_NODES)), (-1, 1)), dtype=torch.float)
        # Add the participant's data to the curvature data list
        curv.append(data)

    return subj, curv


def plot_HCP_stdprocessing_curv(subj_index=0, hemisphere='Left', view_plot=True):
    """
    Plot the curvature map for a given participant from the HCP dataset.
    This method plots curvature values which have been processed with a more
    standard pre-processing pipeline (different to the HCP-specific
    pre-processsing pipeline).

    Args:
        subject_index (int): Index of participant to generate plot for.
                             eg. if subject_index == 0, plot the curvature map
                             for the first participant.
        hemisphere (str): Which hemisphere will be plotted? 'Left' or 'Right'
        view_plot (bool): If true, show the curvature plot as a HTML file
                          in a web browser. If false, don't show the plot
    """
    # Get the list of participants and curvature data
    subj, curv = read_HCP_stdprocessing_curv(hemisphere)

    # Create the background curvature map
    background = np.array(np.reshape(curv[subj_index][0:NUMBER_HEMI_NODES], (-1)))

    threshold = 1 # Threshold for the curvature map

    # Remove NaNs from curvature map
    nocurv = np.isnan(background)
    background[nocurv == 1] = 0
    # Background settings (discretize curvature values to give a 2 colour map)
    background[background < 0] = 0
    background[background > 0] = 1

    # Selecting all visual areas (Wang2015) plus V1-3 fovea
    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
    # Apply ROI mask for the relevant hemisphere
    if hemisphere == 'Left':
        final_mask = final_mask_L
    else:
        # hemisphere == 'Right'
        final_mask = final_mask_R
    
    # Applying masking
    curv_masked = np.zeros((NUMBER_HEMI_NODES, 1))

    # New curvature - Masking - only translating values up
    curv_masked[final_mask == 1] = (np.reshape(np.asarray(
                            curv[subj_index][final_mask == 1]), (-1, 1)) + 1.5)
    curv_masked[final_mask != 1] = 0


    #### Plotting the curvature data ####
    view = plotting.view_surf(
        surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                        'Retinotopy/data/raw/surfaces',
                        f'S1200_7T_Retinotopy181.{hemisphere[0]}' +
                        '.sphere.32k_fs_LR.surf.gii'),
        bg_map=background, surf_map=np.reshape(curv_masked[
        0:NUMBER_HEMI_NODES], (-1)), threshold=threshold, cmap='plasma', 
        black_bg=False, symmetric_cmap=False,
        title=f'HCP participant \
        {subj[subj_index]}: curvature data (standard-processing pipeline, \
        {hemisphere} hemisphere)')

    if view_plot:
        # View plot in web browser
        view.open_in_browser()



plot_HCP_curv(subj_index=0, hemisphere='Left')
plot_HCP_curv(subj_index=0, hemisphere='Right')
plot_HCP_stdprocessing_curv(subj_index=0, hemisphere='Left')
plot_HCP_stdprocessing_curv(subj_index=0, hemisphere='Right')