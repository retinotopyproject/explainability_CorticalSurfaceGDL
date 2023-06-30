import os.path as osp
import scipy.io
import torch

from torch_geometric.data import InMemoryDataset
from Retinotopy.read.read_HCPdata import read_HCP
from Retinotopy.functions.labels import labels
from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi


"""
Used for the HCP dataset, where a standard processing pipeline is applied
(not the processing pipeline used by the HCP study)
Generates the training, dev, and test set separately.
Inherits from InMemoryDataset class (in PyTorch Geometric implementation)
"""
class Retinotopy(InMemoryDataset):
    # Link for the HCP 7T Retinotopy Dataset study
    url = 'https://balsa.wustl.edu/study/show/9Zkk'

    def __init__(self,
                 root,
                 set=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 n_examples=None,
                 myelination=None,
                 prediction=None,
                 hemisphere=None):
        """
        Sets the prediction characteristics (hemisphere and prediction) and
        the feature set (myelination: True or False) for the model. Chooses 
        one of the model's sets.
        Initialises the model's required transform, pre-transform, and filter.
        Sets the file name for the processed data points (for the given 
        model set).
        Args:
            root (string): root directory where the dataset should be saved
            set (string): 'Train', 'Development', or 'Test' set for the model
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
        """
        self.myelination = myelination
        self.prediction = prediction
        self.n_examples = int(n_examples)
        self.hemisphere = hemisphere
        # Super class in PyTorch Geometric implementation
        super(Retinotopy, self).__init__(root, transform, pre_transform,
                                         pre_filter)
        # Set the path to the processed file names for the given set
        self.set = set
        if self.set == 'Train':
            path = self.processed_paths[0]
        elif self.set == 'Development':
            path = self.processed_paths[1]
        else:
            # if self.set == 'Test':
            path = self.processed_paths[2]
        # Load the data and slices from the relevant path
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        """
        Gets the dataset's raw file name.
        Returns:
            (string): the raw file name of the HCP dataset
        """
        return 'S1200_7T_Retinotopy_9Zkk.zip'

    @property
    def processed_file_names(self):
        """
        Gives the processed file names for each of the model's sets, based
        on the characteristics/hemisphere to be used for the predictions,
        as well as the feature set.
        Returns:
            (array of strings): the file names for the processed training, 
            development, and test sets for the selected components for the
            prediction ('Left'/'Right' hemisphere, and 'polarAngle'/
            'eccentricity'/pRF size) and given feature set (including or not 
            including myelination data)
        """
        if self.hemisphere == 'Left':
            if self.myelination == True:
                if self.prediction == 'eccentricity':
                    return [
                        'training_ecc_LH_myelincurv_ROI.pt',
                        'development_ecc_LH_myelincurv_ROI.pt',
                        'test_ecc_LH_myelincurv_ROI.pt']

                elif self.prediction == 'polarAngle':
                    return [
                        'training_PA_LH_myelincurv_ROI.pt',
                        'development_PA_LH_myelincurv_ROI.pt',
                        'test_PA_LH_myelincurv_ROI.pt']

                else:
                    # For pRF size predictions
                    return [
                        'training_pRFsize_LH_myelincurv_ROI.pt',
                        'development_pRFsize_LH_myelincurv_ROI.pt',
                        'test_pRFsize_LH_myelincurv_ROI.pt']
            else:
                # if self.myelination == False:
                if self.prediction == 'eccentricity':
                    return ['training_ecc_LH_curv_ROI.pt',
                            'development_ecc_LH_curv_ROI.pt',
                            'test_ecc_LH_curv_ROI.pt']

                elif self.prediction == 'polarAngle':
                    return ['training_PA_LH_curv_ROI.pt',
                            'development_PA_LH_curv_ROI.pt',
                            'test_PA_LH_curv_ROI.pt']
                else:
                    # For pRF size predictions
                    return [
                        'training_pRFsize_LH_curv_ROI.pt',
                        'development_pRFsize_LH_curv_ROI.pt',
                        'test_pRFsize_LH_curv_ROI.pt']

        else:
            # if self.hemisphere == 'Right':
            if self.myelination == True:
                if self.prediction == 'eccentricity':
                    return [
                        'training_ecc_RH_myelincurv_ROI.pt',
                        'development_ecc_RH_myelincurv_ROI.pt',
                        'test_ecc_RH_myelincurv_ROI.pt']

                elif self.prediction == 'polarAngle':
                    return [
                        'training_PA_RH_myelincurv_ROI.pt',
                        'development_PA_RH_myelincurv_ROI.pt',
                        'test_PA_RH_myelincurv_ROI.pt']

                else:
                    # For pRF size predictions
                    return [
                        'training_pRFsize_RH_myelincurv_ROI.pt',
                        'development_pRFsize_RH_myelincurv_ROI.pt',
                        'test_pRFsize_RH_myelincurv_ROI.pt']
            else:
                # if self.myelination == 'False':
                if self.prediction == 'eccentricity':
                    return ['training_ecc_RH_curv_ROI.pt',
                            'development_ecc_RH_curv_ROI.pt',
                            'test_ecc_RH_curv_ROI.pt']

                elif self.prediction == 'polarAngle':
                    return ['training_PA_RH_curv_ROI.pt',
                            'development_PA_RH_curv_ROI.pt',
                            'test_PA_RH_curv_ROI.pt']

                else:
                    # For pRF size predictions
                    return [
                        'training_pRFsize_RH_curv_ROI.pt',
                        'development_pRFsize_RH_curv_ROI.pt',
                        'test_pRFsize_RH_curv_ROI.pt']


    # def download(self):
    #     raise RuntimeError(
    #         'Dataset not found. Please download S1200_7T_Retinotopy_9Zkk.zip '
    #         'from {} and '
    #         'move it to {} and execute SettingDataset.sh'.format(self.url,
    #                                                              self.raw_dir))


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
            data = read_HCP(path, Hemisphere=self.hemisphere, index=i,
                            surface='mid', visual_mask_L=final_mask_L,
                            visual_mask_R=final_mask_R, faces_L=faces_L,
                            faces_R=faces_R, myelination=self.myelination,
                            prediction=self.prediction)
            if self.pre_transform is not None:
                # Apply a pre-transform to the data
                data = self.pre_transform(data)
            # Add shuffled data point to a list
            data_list.append(data)

        # Split the data points into train, dev, test sets
        train = data_list[0:int(161)]
        dev = data_list[int(161):int(171)]
        test = data_list[int(171):len(data_list)]
        
        # Save each set to .pt file, with the relevant filename
        torch.save(self.collate(train), self.processed_paths[0])
        torch.save(self.collate(dev), self.processed_paths[1])
        torch.save(self.collate(test), self.processed_paths[2])
