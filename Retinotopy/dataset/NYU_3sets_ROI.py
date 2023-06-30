import os.path as osp
import scipy.io
import torch

from torch_geometric.data import InMemoryDataset
from Retinotopy.read.read_NYUdata import read_NYU
from Retinotopy.functions.labels import labels
from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi

"""
Generates the training, dev, and test set separately.
Inherits from InMemoryDataset class (in PyTorch Geometric implementation)
"""
class Retinotopy(InMemoryDataset):
    # Link for the NYU Retinotopy Dataset
    url = 'https://openneuro.org/datasets/ds003787/versions/1.0.0'

    def __init__(self,
                 root,
                 set=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 n_examples=None,
                 myelination=None,
                 prediction=None,
                 hemisphere=None,
                 fine_tuning=False,
                 num_train_subjects=12):
        """
        Sets the prediction characteristics (hemisphere and prediction) and
        the feature set (myelination: True or False) for the model. Chooses 
        one of the model's sets.
        Initialises the model's required transform, pre-transform, and filter.
        Sets the file name for the processed data points (for the given 
        model set).
        Args:
            root (string): root directory where the dataset should be saved
            set (string): 'Train', or 'Test' set for the model
                          (parsing 'Development' will throw an exception)
            transform (callable): a function/transform that takes in an :obj:
                                  `torch_geometric.data.Data` object and 
                                  returns a transformed version. The data 
                                  object will be transformed before every 
                                  access.
            pre_transform (callable): A function/transform that takes in an 
                                      :obj:`torch_geometric.data.Data` object 
                                      and returns a transformed version. The 
                                      data object will be transformed before 
                                      being saved to disk.
            pre_filter (callable): A function that takes in an :obj:
                                  `torch_geometric.data.Data` object and 
                                  returns a boolean value, indicating whether 
                                  the data object should be included in the 
                                  final dataset.
            n_examples (int): the number of participants to be loaded from the
                              dataset (HCP has 181 total participants)
            myelination (boolean): if True, use myelination in the feature set;
                                   if False, use only curvature as a feature
            prediction (string): 'polarAngle' or 'eccentricity'. If another
                                  value is provided, pRF size will be predicted
            hemisphere (string): 'Left' or 'Right' hemisphere
            fine_tuning (boolean): if True, finetuning will be applied to the
                                   already-trained model (a train set 
                                   containing datapoints for finetuning will be
                                   created). If False, only a test set will be 
                                   generated.
            num_train_subjects (int): number of subjects to add to the training 
                                      set for fine-tuning (12 by default). If 
                                      fine_tuning == False, value is ignored.
        """
        self.myelination = myelination
        self.prediction = prediction
        self.n_examples = int(n_examples)
        self.hemisphere = hemisphere
        self.fine_tuning = fine_tuning
        self.num_train_subjects = int(num_train_subjects)
        # Super class in PyTorch Geometric implementation
        super(Retinotopy, self).__init__(root, transform, pre_transform,
                                         pre_filter)
        # Set the path to the processed file names for the given set
        self.set = set
        if self.set == 'Train':
            path = self.processed_paths[0]
        elif self.set == 'Development':
            '''
            Dev set has not been configured here - if you want to create a dev
            set, you will need to modify process() function to allocate some
            subjects to a dev set, then save this using torch.save().
            Providing 'Development' arg will raise an exception to prevent
            a dev set from being incorrectly instantiated. You can comment
            this exception out if you want to add a dev set.
            '''
            path = self.processed_paths[1]
            raise Exception('NYU dataset has not been configured to' +
            'generate development sets. Please modify the file' +
            'NYU_3sets_ROI.py to be able to allocate subjects into a dev set.')
        else:
            # if self.set == 'Test':
            path = self.processed_paths[2]
        # Load the data and slices from the relevant path
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        """
        This method was used in HCP_3sets_ROI.py to get the dataset's raw
        file name. It isn't used here, but I'm leaving this method here
        in case removing it breaks something
        Returns:
            (string): the raw file name of the HCP dataset
        """
        return 'S1200_7T_Retinotopy_9Zkk.zip'

    @property
    def processed_file_names(self):
        """
        Gives the processed file names for each of the model's sets, based
        on the characteristics/hemisphere to be used for the predictions,
        as well as the feature set. Filenames will also include the number
        of participants used for finetuning the model (if finetuning is
        used).
        Returns:
            (array of strings): the file names for the processed training, 
            development, and test sets for the selected components for the
            prediction ('Left'/'Right' hemisphere, and 'polarAngle'/
            'eccentricity'/pRF size) and given feature set (including or not 
            including myelination data), and number of participants used in
            finetuning.
        """
        # Add additional info to filenames if fine-tuning is being used
        fine_tuning_filename = ""
        if self.fine_tuning:
            # Add the number of subjects used to fine-tune to the filename
            fine_tuning_filename = f'fineTuning_{self.num_train_subjects}subj_'
        
        if self.hemisphere == 'Left':
            if self.myelination == True:
                if self.prediction == 'eccentricity':
                    return [
                        f'NYU_{fine_tuning_filename}training_ecc_LH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}development_ecc_LH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}test_ecc_LH_myelincurv_ROI.pt']

                elif self.prediction == 'polarAngle':
                    return [
                        f'NYU_{fine_tuning_filename}training_PA_LH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}development_PA_LH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}test_PA_LH_myelincurv_ROI.pt']

                else:
                    # For pRF size predictions (not used in the project)
                    return [
                        f'NYU_{fine_tuning_filename}training_pRFsize_LH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}development_pRFsize_LH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}test_pRFsize_LH_myelincurv_ROI.pt']
            else:
                # if self.myelination == False:
                if self.prediction == 'eccentricity':
                    return [f'NYU_{fine_tuning_filename}training_ecc_LH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}development_ecc_LH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}test_ecc_LH_curv_ROI.pt']

                elif self.prediction == 'polarAngle':
                    return [f'NYU_{fine_tuning_filename}training_PA_LH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}development_PA_LH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}test_PA_LH_curv_ROI.pt']
                else:
                    # For pRF size predictions (not used in the project)
                    return [
                        f'NYU_{fine_tuning_filename}training_pRFsize_LH_curv_ROI.pt',
                        f'NYU_{fine_tuning_filename}development_pRFsize_LH_curv_ROI.pt',
                        f'NYU_{fine_tuning_filename}test_pRFsize_LH_curv_ROI.pt']

        else:
            # if self.hemisphere == 'Right':
            if self.myelination == True:
                if self.prediction == 'eccentricity':
                    return [
                        f'NYU_{fine_tuning_filename}training_ecc_RH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}development_ecc_RH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}test_ecc_RH_myelincurv_ROI.pt']

                elif self.prediction == 'polarAngle':
                    return [
                        f'NYU_{fine_tuning_filename}training_PA_RH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}development_PA_RH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}test_PA_RH_myelincurv_ROI.pt']

                else:
                    # For pRF size predictions (not used in the project)
                    return [
                        f'NYU_{fine_tuning_filename}training_pRFsize_RH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}development_pRFsize_RH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}test_pRFsize_RH_myelincurv_ROI.pt']
            else:
                # if self.myelination == 'False':
                if self.prediction == 'eccentricity':
                    return [f'NYU_{fine_tuning_filename}training_ecc_RH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}development_ecc_RH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}test_ecc_RH_curv_ROI.pt']

                elif self.prediction == 'polarAngle':
                    return [f'NYU_{fine_tuning_filename}training_PA_RH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}development_PA_RH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}test_PA_RH_curv_ROI.pt']

                else:
                    # For pRF size predictions (not used in the project)
                    return [
                        f'NYU_{fine_tuning_filename}training_pRFsize_RH_curv_ROI.pt',
                        f'NYU_{fine_tuning_filename}development_pRFsize_RH_curv_ROI.pt',
                        f'NYU_{fine_tuning_filename}test_pRFsize_RH_curv_ROI.pt']

    def process(self):
        """
        Reads the data for each participant in the HCP dataset.
        Splits the data points into Train, Development, and Test sets.
        Saves each set as a .pt file.
        Note: Shuffling of the data points should occur within the
              read_HCP method.
        """
        path = osp.join(self.raw_dir, 'converted')
        data_list = []

        # Selecting all visual areas (Wang2015) plus V1-3 fovea
        label_primary_visual_areas = ['ROI']
        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
            label_primary_visual_areas)
        faces_R = labels(scipy.io.loadmat(osp.join(path, 'tri_faces_R.mat'))[
                             'tri_faces_R'] - 1, index_R_mask)
        faces_L = labels(scipy.io.loadmat(osp.join(path, 'tri_faces_L.mat'))[
                             'tri_faces_L'] - 1, index_L_mask)

        # Load each participant from the dataset
        for i in range(0, self.n_examples):
            data = read_NYU(path, Hemisphere=self.hemisphere, index=i,
                            surface='mid', visual_mask_L=final_mask_L,
                            visual_mask_R=final_mask_R, faces_L=faces_L,
                            faces_R=faces_R, prediction=self.prediction)
            if self.pre_transform is not None:
                # Apply a pre-transform to the data
                data = self.pre_transform(data)
            # Add shuffled data point to a list
            data_list.append(data)

        if self.fine_tuning:
            # Generate a train set of size (num_train_subjects)
            train = data_list[0:self.num_train_subjects]
            # Test set contains the remaining participants
            test = data_list[self.num_train_subjects:len(data_list)]
            # Save sets
            torch.save(self.collate(train), self.processed_paths[0])
            torch.save(self.collate(test), self.processed_paths[2])
        else:
            # (No finetuning) Generate only a test set
            test = data_list[0:len(data_list)]
            # Save set
            torch.save(self.collate(test), self.processed_paths[2])
        
