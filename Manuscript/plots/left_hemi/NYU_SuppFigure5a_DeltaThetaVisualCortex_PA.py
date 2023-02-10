import numpy as np
import scipy.io
import os.path as osp
import torch

# For loading new curvature data
import nibabel as nib

from nilearn import plotting
from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from Retinotopy.functions.error_metrics import smallest_angle

# path = './../../../Retinotopy/data/raw/converted'
path = './../../../Retinotopy/data/nyu_converted'
# curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
# background = np.reshape(curv['x100610_curvature'][0][0][0:32492], (-1))

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
subject_index = 0
background = np.array(np.reshape(curv[subject_index][0:32492], (-1)))

threshold = 1

nocurv = np.isnan(background)
background[nocurv == 1] = 0

# Predictions generated with 4 sets of features (pred = intact features)
# models = ['pred', 'rotatedROI', 'shuffled-myelincurv', 'constant']
num_models = 5

mean_delta = [] # Prediction error
mean_across = [] # Individual variability

num_epochs = 5
for m in range(1, num_models+1):
    predictions = torch.load(osp.join('./../../../Models/generalizability',
        'NYU_testset_fineTuned_results', 
        f'NYU_testset_fineTuned_{num_epochs}epochs_-intactData_PA_LH_model' + 
        str(m) + '.pt'), map_location='cpu')

    theta_withinsubj = []
    theta_acrosssubj_pred = []

    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
        label_primary_visual_areas)
    ROI1 = np.zeros((32492, 1))
    ROI1[final_mask_L == 1] = 1

    mask = ROI1
    mask = mask[ROI1 == 1]

    # Compute angle between predicted and empirical predictions across subj
    for j in range(len(predictions['Predicted_values'])):
        theta_pred_across_temp = []

        for i in range(len(predictions['Predicted_values'])):
            # Compute the difference between predicted and empirical angles
            # within subj - error
            if i == j:
                # Loading predicted values
                pred = np.reshape(np.array(predictions['Predicted_values'][i]),
                                  (-1, 1))
                measured = np.reshape(
                    np.array(predictions['Measured_values'][j]),
                    (-1, 1))


                # Rescaling polar angles to match the correct visual field (
                # left hemisphere)
                minus = pred > 180
                sum = pred < 180
                pred[minus] = pred[minus] - 180
                pred[sum] = pred[sum] + 180
                pred = np.array(pred) * (np.pi / 180)

                minus = measured > 180
                sum = measured < 180
                measured[minus] = measured[minus] - 180
                measured[sum] = measured[sum] + 180
                measured = np.array(measured) * (np.pi / 180)

                # Computing delta theta, difference between predicted and
                # empirical angles
                theta = smallest_angle(pred, measured)
                theta_withinsubj.append(theta)

            if i != j:
                # Compute the difference between predicted maps
                # across subj - individual variability

                # Loading predicted values
                pred = np.reshape(np.array(predictions['Predicted_values'][i]),
                                  (-1, 1))
                pred2 = np.reshape(
                    np.array(predictions['Predicted_values'][j]), (-1, 1))

                # Rescaling polar angles to match the correct visual field (
                # left hemisphere)
                minus = pred > 180
                sum = pred < 180
                pred[minus] = pred[minus] - 180
                pred[sum] = pred[sum] + 180
                pred = np.array(pred) * (np.pi / 180)

                minus = pred2 > 180
                sum = pred2 < 180
                pred2[minus] = pred2[minus] - 180
                pred2[sum] = pred2[sum] + 180
                pred2 = np.array(pred2) * (np.pi / 180)

                # Computing delta theta, difference between predicted maps
                theta_pred = smallest_angle(pred, pred2)
                theta_pred_across_temp.append(theta_pred)

        theta_acrosssubj_pred.append(np.mean(theta_pred_across_temp, axis=0))

    mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)
    mean_theta_acrosssubj_pred = np.mean(np.array(theta_acrosssubj_pred),
                                         axis=0)

    mean_delta.append(mean_theta_withinsubj[mask == 1])
    mean_across.append(mean_theta_acrosssubj_pred[mask == 1])

mean_delta = np.reshape(np.array(mean_delta), (num_models, -1))
mean_across = np.reshape(np.array(mean_across), (num_models, -1))

# Generating plots
# Select predictions generated with a given set of features
# model_index = np.where(np.array(models) == 'rotatedROI')
# Selecting predictions generated with model 5
model_index = 4

# Region of interest
delta_theta = np.ones((32492, 1))
delta_theta[final_mask_L == 1] = np.reshape(mean_delta[model_index],
                                            (3267, 1)) + threshold
delta_theta[final_mask_L != 1] = 0

delta_across = np.ones((32492, 1))
delta_across[final_mask_L == 1] = np.reshape(mean_across[model_index],
                                             (3267, 1)) + threshold
delta_across[final_mask_L != 1] = 0

# Error map
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(delta_theta[0:32492], (-1)), bg_map=background,
    cmap='Reds', black_bg=False, symmetric_cmap=False, threshold=threshold,
    vmax=75 + threshold,
    title=f'Prediction error - {num_epochs} epochs')
view.open_in_browser()

# Individual variability map
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(delta_across[0:32492], (-1)), bg_map=background,
    cmap='Blues', black_bg=False, symmetric_cmap=False, threshold=threshold,
    vmax=75 + threshold,
    title=f'Individual variability = {num_epochs} epochs')
view.open_in_browser()