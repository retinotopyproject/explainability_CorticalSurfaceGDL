import numpy as np
import scipy.io
import os.path as osp
import torch

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from Retinotopy.functions.def_ROIs_DorsalEarlyVisualCortex import roi as roi2
from nilearn import plotting

# For loading new curvature data
import nibabel as nib

# subject_index = 7

# hcp_id = ['617748', '191336', '572045', '725751', '198653',
#           '601127', '644246', '191841', '680957', '157336']

# cd Manuscript/plots/left_hemi
# path = './../../../Retinotopy/data/raw/converted'
path = './../../../Retinotopy/data/nyu_converted'

# For loading old curvature data
# curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
# background = np.reshape(
#     curv['x' + hcp_id[subject_index] + '_curvature'][0][0][0:32492], (-1))


# For NYU curvature data:
# Loading subject IDs (in the order in which they were selected for train, dev, test datasets)
with open(osp.join(path, '../..', 'participant_IDs_in_order.txt')) as fp:
    subjects = fp.read().split("\n")
subjects = subjects[0:len(subjects) - 1]
# with open(osp.join(path, '../..', 'list_subj')) as fp:
#     subjects = fp.read().split("\n")
# subjects = subjects[0:len(subjects) - 1]

# Defining number of nodes
number_cortical_nodes = int(64984)
number_hemi_nodes = int(number_cortical_nodes / 2)

curv = []
# Index of the first subject in the testing dataset
test_index_start = 0
# test_index_start = 12
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
subject_index = 12
# for subject_index in range(0, 10):
background = np.array(np.reshape(curv[subject_index][0:32492], (-1)))

threshold = 1  # threshold for the curvature map

# Background settings
nocurv = np.isnan(background)
background[nocurv == 1] = 0

# Use these to modify background into discrete colour regions - but you will have to tweak the params so 
# it doesn't look bad for the new curv data!
# background[background > 0] = 0.3
# background[background < 0] = 0.5
# background[background < 0] = 0
# background[background > 0] = 1


# ROI settings
label_primary_visual_areas = ['ROI']
final_mask_L_ROI, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
ROI_masked = np.zeros((32492, 1))
ROI_masked[final_mask_L_ROI == 1] = 1

pred = np.zeros((32492, 1))
measured = np.zeros((32492, 1))

# Dorsal V1-V3
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(['ROI'])
dorsal_earlyVisualCortex = np.zeros((32492, 1))
dorsal_earlyVisualCortex[final_mask_L == 1] = 1

# # Final mask (selecting dorsal V1-V3 vertices)
# mask = ROI_masked + dorsal_earlyVisualCortex
# mask = mask[ROI_masked == 1]
# # print(np.shape(mask))
# pred[final_mask_L_ROI == 1] = np.reshape(mask,(-1,1))

# mask = mask[ROI_masked == 1]


# Loading the predictions
# predictions = torch.load(
#     './../../testset_results/left_hemi'
#     '/testset-pred_deepRetinotopy_PA_LH.pt',
#     map_location='cpu')
# selected_model = 1
num_epochs = 5
for selected_model in range(1, 6):
    # predictions = torch.load('./../../../Models/generalizability/NYU_testset_results/'
    #     '/NYU_testset-intactData_PA_LH_model' + str(selected_model) + '.pt',
    #     map_location='cpu')
    predictions = torch.load('./../../../Models/generalizability/NYU_testset_fineTuned_results/'
        f'/NYU_testset_fineTuned_{num_epochs}epochs_-intactData_PA_LH_model' + str(selected_model) + '.pt',
        map_location='cpu')

    # pred[final_mask_L == 1] = np.reshape(
    #     np.array(predictions['Predicted_values'][subject_index]),
    #     (-1, 1))

    pred[final_mask_L_ROI == 1] = np.reshape(
        np.array(predictions['Predicted_values'][subject_index]),
        (-1, 1))

    # measured[final_mask_L == 1] = np.reshape(
    #     np.array(predictions['Measured_values'][subject_index]),
    #     (-1, 1))

    measured[final_mask_L_ROI == 1] = np.reshape(
        np.array(predictions['Measured_values'][subject_index]),
        (-1, 1))

    # Rescaling
    pred = np.array(pred)
    minus = pred > 180
    sum = pred < 180
    pred[minus] = pred[minus] - 180 + threshold
    pred[sum] = pred[sum] + 180 + threshold
    pred = np.array(pred)

    '''
    Transforms data in readHCP, then transform it back here
    '''
    measured = np.array(measured)
    minus = measured > 180
    sum = measured < 180
    measured[minus] = measured[minus] - 180 + threshold
    measured[sum] = measured[sum] + 180 + threshold
    measured = np.array(measured)

    # List of nodes
    # kernel = np.load('./../../Models/10hops_neighbors_test.npz')['list']
    # kernel = np.load('/home/uqfribe1/PycharmProjects/deepRetinotopy_explain'
    #                  '/Models/nodes_earlyVisualCortex.npz')['list']
    # kernel = np.load('./../../../Models/nodes_earlyVisualCortex.npz')['list']
    # kernel = [1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726,
    #        1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767,
    #        1768, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797,
    #        1798, 1799, 1800, 1801, 1814, 1815, 1816, 1817, 1818, 1819, 1820,
    #        1821, 1822, 1823, 1824, 1825, 1826, 1827, 1837, 1838, 1839, 1840,
    #        1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851,
    #        1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868,
    #        1869, 1870, 1871, 1872, 1873, 1877, 1878, 1879, 1880, 1881, 1882,
    #        1883, 1884]
    # transform_kernel = np.where(final_mask_L==1)[0][kernel]

    # # Neighborhood
    # new_pred = np.zeros(np.shape(pred))
    # for i in range(len(pred)):
    #     if np.sum(transform_kernel==i)!=0:
    #         new_pred[i][0] = 0
    #         # print(new_pred[i][0])
    #     else:
    #         new_pred[i][0] = pred[i][0]

    # Masking
    # measured[final_mask_L != 1] = 0
    measured[final_mask_L_ROI != 1] = 0 # removes data from outside ROI
    pred[pred == 1] = 100
    pred[pred == 2] = 2
    pred[final_mask_L_ROI != 1] = 0 # removes data from outside ROI


    # print(len(np.where(mask == 2)[0]))
    # new_pred[final_mask_L != 1] = 0

    # Predicted map
    view = plotting.view_surf(
        # surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
        #                    'Retinotopy/data/raw/surfaces'
        #                    '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
        # surf_map=np.reshape(pred[0:32492], (-1)), bg_map=background,
        # cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
        # threshold=threshold, vmax=361)
        surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                        'Retinotopy/data/raw/surfaces'
                        '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
        bg_map=background, surf_map=np.reshape(pred[0:32492], (-1)),
        cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
        threshold=threshold, vmax=361,
        title=f'(NYU) Participant {subject_index+1}: Polar angle Left hemisphere predictions (Model {selected_model})')
    view.open_in_browser()

    # view.save_as_html(osp.join('D:\\Retinotopy Project\\surf_images\\NYU_originalCurv\\PA_LH\\', f'predicted_model{selected_model}_participant{subject_index+1}'))

# Empirical map
# view = plotting.view_surf(
#     surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
#                        'Retinotopy/data/raw/surfaces'
#                        '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
#     surf_map=np.reshape(measured[0:32492], (-1)), bg_map=background,
#     cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
#     threshold=threshold, vmax=361)
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                    'Retinotopy/data/raw/surfaces'
                    '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    bg_map=background, surf_map=np.reshape(measured[0:32492], (-1)),
    cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=361,
    title=f'(NYU) Participant {subject_index+1}: Polar angle Left hemisphere ground truth')
view.open_in_browser()
# view.save_as_html(f'D:\\Retinotopy Project\\surf_images\\NYU_originalCurv\\PA_LH\\empirical_participant{subject_index+1}')