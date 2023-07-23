import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import sys
import os

"""
This code was copied from the deepRetinotopy repository, from the file 
'Figure7b_DeltaTheta_ModelvsAverageMap.py' in the Manuscript/plots/left_hemi 
dir (https://github.com/Puckett-Lab/deepRetinotopy/)

The code generates a point plot of prediction error based on the model 
test set predictions and an average-based prediction. Plots can be generated for 
the Left or Right hemisphere, for either polar angle or eccentricity values,
for participants in either the HCP test set or the NYU test set. It can be 
used with either finetuned or non-finetuned models that use the NYU dataset. 

To generate this plot, several other files must be generated first by 
running either ModelEval_MeanDeltaTheta_ECC.py (for eccentricity LH or RH),
or ModelEval_MeanDelta_Theta_PA.py (for polar angle LH or RH). Certain other
files are also required for these other .py files to run correctly -
see the docstrings in the ModelEval files for more details about the 
requirements.

Note: code implementation assumes that the file is being run from the dir 
Manuscript/plots - I have modified the code to automatically set the 
working dir to this (if it isn't already).
"""
# Set the working directory to Manuscript/plots
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Configure filepaths
sys.path.append('../..')


# Number of test set participants in HCP dataset
HCP_TEST_SET_SIZE = int(10)
# Total number of participants in NYU dataset
NYU_N_EXAMPLES = int(43)


def error_plots(hemisphere, retinotopic_feature, selected_model=1, 
                dataset='HCP', n_finetuning_subj=None, num_epochs=20):
    """Function to generate error plot.

    Args:
        hemisphere (str): 'LH' or 'RH'.
        retinotopic_feature (str): 'PA' or 'ECC' or 'pRFcenter'.
        dataset (str): for which dataset will plots be created?
                       ('HCP' or 'NYU')
        n_finetuning_subj (str or None): How many participants were 
                           allocated to a 'Training' set for finetuning?
                           If n_finetuning_subj == None, finetuning 
                           was not performed. This var is ignored if
                           dataset == 'HCP'.
        num_epochs (int): How many epochs did finetuning occur for? If 
                          n_finetuning_subj == None or dataset == 'HCP',
                          the var is ignored (as finetuning didn't take place).

    Returns:
        Point plot of prediction error from our models' predictions and an
        average-based prediction.
    """
    # Create the file name components and test set size for the given params
    hemi_filename, dataset_filename, testset_size, testset_results_dir = \
                        set_filenames_and_testset_size(hemisphere=hemisphere,
                            dataset=dataset, n_finetuning_subj=n_finetuning_subj, 
                            num_epochs=num_epochs)

    # Load the error per participant files
    error_DorsalEarlyVisualCortex_model = np.reshape(np.array(
        np.load(f'./../stats/output/{dataset_filename}ErrorPerParticipant_' + 
            f'{str(retinotopic_feature)}_{hemi_filename}H_' +
            f'model{str(selected_model)}_dorsalV1-3_' +
            'deepRetinotopy_1-8.npz')['list']), (testset_size, -1))
    error_EarlyVisualCortex_model = np.reshape(np.array(
        np.load(f'./../stats/output/{dataset_filename}ErrorPerParticipant_' + 
            f'{str(retinotopic_feature)}_{hemi_filename}H_' +
            f'model{str(selected_model)}_EarlyVisualCortex_' +
            'deepRetinotopy_1-8.npz')['list']), (testset_size, -1))
    error_higherOrder_model = np.reshape(np.array(
        np.load(f'./../stats/output/{dataset_filename}ErrorPerParticipant_' + 
            f'{str(retinotopic_feature)}_{hemi_filename}H_' +
            f'model{str(selected_model)}_WangParcels_' +
            'deepRetinotopy_1-8.npz')['list']), (testset_size, -1))

    error_DorsalEarlyVisualCortex_average = np.reshape(np.array(
        np.load(f'./../stats/output/{dataset_filename}ErrorPerParticipant_' + 
            f'{str(retinotopic_feature)}_{hemi_filename}H_' +
            f'model{str(selected_model)}_dorsalV1-3_' +
            'average_1-8.npz')['list']), (testset_size, -1))
    error_EarlyVisualCortex_average = np.reshape(np.array(
        np.load(f'./../stats/output/{dataset_filename}ErrorPerParticipant_' + 
            f'{str(retinotopic_feature)}_{hemi_filename}H_' +
            f'model{str(selected_model)}_EarlyVisualCortex_' +
            'average_1-8.npz')['list']), (testset_size, -1))
    error_higherOrder_average = np.reshape(np.array(
        np.load(f'./../stats/output/{dataset_filename}ErrorPerParticipant_' + 
            f'{str(retinotopic_feature)}_{hemi_filename}H_' +
            f'model{str(selected_model)}_WangParcels_' +
            'average_1-8.npz')['list']), (testset_size, -1))

    # Reformatting data from dorsal early visual cortex
    data_earlyVisualCortexDorsal = np.concatenate([
        [np.mean(error_DorsalEarlyVisualCortex_average, axis=1),
         np.shape(error_DorsalEarlyVisualCortex_average)[0] * [
             'Average Map'],
         np.shape(error_DorsalEarlyVisualCortex_average)[0] * [
             'Early Visual Cortex']
         ],
        [np.mean(error_DorsalEarlyVisualCortex_model, axis=1),
         np.shape(error_DorsalEarlyVisualCortex_model)[0] * ['Model'],
         np.shape(error_DorsalEarlyVisualCortex_model)[0] * [
             'Early Visual Cortex']]],
        axis=1)

    df_0 = pd.DataFrame(
        columns=['$\Delta$$\t\Theta$', 'Prediction', 'Area'],
        data=data_earlyVisualCortexDorsal.T)
    df_0['$\Delta$$\t\Theta$'] = df_0['$\Delta$$\t\Theta$'].astype(float)

    print(
        scipy.stats.ttest_rel(
            np.mean(error_DorsalEarlyVisualCortex_model, axis=1),
            np.mean(error_DorsalEarlyVisualCortex_average,
                    axis=1)))  # Repeated measures t-test

    # Reformatting data from early visual cortex
    data_earlyVisualCortex = np.concatenate([
        [np.mean(error_EarlyVisualCortex_average, axis=1),
         np.shape(error_EarlyVisualCortex_average)[0] * [
             'Average Map'],
         np.shape(error_EarlyVisualCortex_average)[0] * [
             'Early Visual Cortex']
         ],
        [np.mean(error_EarlyVisualCortex_model, axis=1),
         np.shape(error_EarlyVisualCortex_model)[0] * ['Model'],
         np.shape(error_EarlyVisualCortex_model)[0] * [
             'Early Visual Cortex']]],
        axis=1)

    df_1 = pd.DataFrame(
        columns=['$\Delta$$\t\Theta$', 'Prediction', 'Area'],
        data=data_earlyVisualCortex.T)
    df_1['$\Delta$$\t\Theta$'] = df_1['$\Delta$$\t\Theta$'].astype(float)

    print(scipy.stats.ttest_rel(np.mean(error_EarlyVisualCortex_model, axis=1),
                                np.mean(error_EarlyVisualCortex_average,
                                        axis=1)))  # Repeated measures t-test

    # Reformatting data from higher order visual areas
    data_HigherOrder = np.concatenate([
        [np.mean(error_higherOrder_average, axis=1),
         np.shape(error_higherOrder_average)[0] * [
             'Average Map'],
         np.shape(error_higherOrder_average)[0] * [
             'Higher Order Visual Areas']],
        [np.mean(error_higherOrder_model, axis=1),
         np.shape(error_higherOrder_model)[0] * ['Model'],
         np.shape(error_higherOrder_model)[0] * [
             'Higher Order Visual Areas']]
    ],
        axis=1)

    df_2 = pd.DataFrame(
        columns=['$\Delta$$\t\Theta$', 'Prediction', 'Area'],
        data=data_HigherOrder.T)
    df_2['$\Delta$$\t\Theta$'] = df_2['$\Delta$$\t\Theta$'].astype(float)

    print(scipy.stats.ttest_rel(np.mean(error_higherOrder_model, axis=1),
                                np.mean(error_higherOrder_average,
                                        axis=1)))  # Repeated measures t-test

    # Generate the plot
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    title = ['Dorsal V1-3', 'Early visual cortex', 'Higher order visual areas']
    fig = plt.figure(figsize=(10, 5))
    palette = sns.color_palette("Paired")[4:6][::-1]
    for i in range(3):
        fig.add_subplot(1, 3, i + 1)
        ax = sns.swarmplot(x='Prediction', y='$\Delta$$\t\Theta$',
                           data=eval('df_' + str(i)),
                           palette=palette)
        plt.title(title[i])

        # Prediction error for the same participants
        for j in range(testset_size):
            x = [eval('df_' + str(i))['Prediction'][j],
                 eval('df_' + str(i))['Prediction'][j + testset_size]]
            y = [eval('df_' + str(i))['$\Delta$$\t\Theta$'][j],
                 eval('df_' + str(i))['$\Delta$$\t\Theta$'][j + testset_size]]
            ax.plot(x, y, color='black', alpha=0.1)
            # Rescale Dorsal/Early Visual Cortex axes differently per dataset
            if retinotopic_feature == 'PA':
                plt.ylim([10, 35])  # HCP scaling (default)
                if dataset == 'NYU':
                    plt.ylim([10, 70])  # NYU scaling
            if retinotopic_feature == 'ECC':
                plt.ylim([0, 2])    # HCP scaling (default)
                if dataset == 'NYU':
                    plt.ylim([1.5, 10])    # NYU scaling
            if retinotopic_feature == 'pRFcenter':
                plt.ylim([0, 1])
            # plt.ylim([15, 45])
    # Rescale Higher Order Visual Areas y axes differently for each dataset
    if retinotopic_feature == 'PA':
        plt.ylim([30, 75])  # HCP scaling (default)
        if dataset == 'NYU':
            plt.ylim([15, 110])  # NYU scaling
    if retinotopic_feature == 'ECC':
        plt.ylim([1.5, 4])  # HCP scaling (default)
        if dataset == 'NYU':
            plt.ylim([4, 15]) # NYU scaling
    if retinotopic_feature == 'pRFcenter':
        plt.ylim([0, 3])

    # plt.savefig('./output/DeltaTheta_ModelvsAverage.pdf', format="pdf")
    return plt.show()


def set_filenames_and_testset_size(hemisphere, dataset, n_finetuning_subj,
                                    num_epochs):
    """
    Helper method, used to assign file name components (hemisphere file name,
    test set filename, test set directory) and the size of the test set for the 
    given dataset.
    Args:
        hemisphere (str): For which hemisphere will files be generated?
                          ('Left' or 'Right')
        dataset (str): For which of the datasets used will files be generated?
                          ('HCP' or 'NYU')
        n_finetuning_subj (str or None): How many participants were 
                           allocated to a 'Training' set for finetuning?
                           If n_finetuning_subj == None, finetuning 
                           was not performed. This var is ignored if
                           dataset == 'HCP'.
        num_epochs (int): How many epochs did finetuning occur for? If 
                          n_finetuning_subj == None or dataset == 'HCP',
                          the var is ignored (as finetuning didn't take place).

    Output:
        hemi_filename (str): Shortened version of hemisphere string. 'L' or 'R'
        dataset_filename (str): includes the dataset name, as well as the 
                                number of finetuning participants and number
                                of finetuning epochs (if finetuning was 
                                performed)
        testset_size (int): the number of participants in the test set
        testset_results_dir (str): the directory in which the test set results
                                   are saved ('testset_results', 'NYU_testset_results', 
                                   'NYU_testset_finetuned_results')

    """
    # Create the file name components for the chosen hemisphere
    hemi_filename = hemisphere[0]

    # Set the file names and number of test set participants (based on the dataset used)
    if dataset == 'HCP':
        dataset_filename = ''
        # 10 participants in the test set
        testset_size = HCP_TEST_SET_SIZE
        # Set the name of the testset results directory
        testset_results_dir = 'testset_results'

    elif dataset == 'NYU':
        # If no finetuning:
        dataset_filename = dataset
        # 43 participants in the test set
        testset_size = NYU_N_EXAMPLES
        # Set the name of the testset results directory
        testset_results_dir = 'NYU_testset_results'

        # If finetuning is performed:
        if n_finetuning_subj is not None:
            # Add the number of subjects used to finetune and number of epochs
            dataset_filename += \
                f'_finetuned_{n_finetuning_subj}subj_{num_epochs}epochs'
            # Remove the # of finetuning participants from the test set size
            testset_size -= n_finetuning_subj
            # Add 'finetuned' to the testset results dir name if required
            testset_results_dir = 'NYU_testset_finetuned_results'
    
        # Add a trailing '_' to dataset filename
        dataset_filename += '_'
    
    return hemi_filename, dataset_filename, testset_size, testset_results_dir


# # Generates files for HCP models
# for hemi in ['Left', 'Right']:
#     # Polar angle and ecc
#     for feature in ['PA', 'ECC']:
#         error_plots(hemisphere=hemi, selected_model=5, dataset='HCP',
#                     retinotopic_feature=feature)

# # Generates files for NYU models (no finetuning)
# for hemi in ['Left', 'Right']:
#     # Polar angle and ecc
#     for feature in ['PA', 'ECC']:
#         error_plots(hemisphere=hemi, selected_model=5, dataset='NYU',
#                     retinotopic_feature=feature)

# # Generates files for NYU models with finetuning
# for hemi in ['Left', 'Right']:
#     # Polar angle and ecc
#     for feature in ['PA', 'ECC']:
#         # Finetuning sets with 8 and 12 participants
#         for n_subj in [8, 12]:
#             # Finetuning for 5, 10, 20 epochs
#             for epochs in [5, 10, 20]:
#                 error_plots(hemisphere=hemi, selected_model=5, dataset='NYU', 
#                             retinotopic_feature=feature,
#                             n_finetuning_subj=n_subj, num_epochs=epochs)

# pRF center was not used as a retinotopic feature for this project
# error_plots('LH', 'pRFcenter')