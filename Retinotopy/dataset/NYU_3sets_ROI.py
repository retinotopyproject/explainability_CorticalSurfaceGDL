import os.path as osp
import scipy.io
import torch

from torch_geometric.data import InMemoryDataset
from Retinotopy.read.read_NYUdata import read_NYU
from Retinotopy.functions.labels import labels
from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi


# Generates the training, dev and test sets separately


class Retinotopy(InMemoryDataset):
    url = 'https://balsa.wustl.edu/study/show/9Zkk'
    # TODO replace this URL

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
        self.myelination = myelination
        self.prediction = prediction
        self.n_examples = int(n_examples)
        self.hemisphere = hemisphere

        # Will fine-tuning be applied to the original training model?
        self.fine_tuning = fine_tuning
        '''
        Number of subjects to add to the training set for fine-tuning (12 by 
        default). If fine_tuning == False, this value is ignored, and only a
        test set is generated.
        '''
        self.num_train_subjects = int(num_train_subjects)

        super(Retinotopy, self).__init__(root, transform, pre_transform,
                                         pre_filter)
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
            path = self.processed_paths[2]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return 'S1200_7T_Retinotopy_9Zkk.zip'

    @property
    def processed_file_names(self):
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
                    return [
                        f'NYU_{fine_tuning_filename}training_pRFsize_LH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}development_pRFsize_LH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}test_pRFsize_LH_myelincurv_ROI.pt']
            else:
                if self.prediction == 'eccentricity':
                    return [f'NYU_{fine_tuning_filename}training_ecc_LH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}development_ecc_LH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}test_ecc_LH_curv_ROI.pt']

                elif self.prediction == 'polarAngle':
                    return [f'NYU_{fine_tuning_filename}training_PA_LH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}development_PA_LH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}test_PA_LH_curv_ROI.pt']
                else:
                    return [
                        f'NYU_{fine_tuning_filename}training_pRFsize_LH_curv_ROI.pt',
                        f'NYU_{fine_tuning_filename}development_pRFsize_LH_curv_ROI.pt',
                        f'NYU_{fine_tuning_filename}test_pRFsize_LH_curv_ROI.pt']

        else:
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
                    return [
                        f'NYU_{fine_tuning_filename}training_pRFsize_RH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}development_pRFsize_RH_myelincurv_ROI.pt',
                        f'NYU_{fine_tuning_filename}test_pRFsize_RH_myelincurv_ROI.pt']
            else:
                if self.prediction == 'eccentricity':
                    return [f'NYU_{fine_tuning_filename}training_ecc_RH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}development_ecc_RH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}test_ecc_RH_curv_ROI.pt']

                elif self.prediction == 'polarAngle':
                    return [f'NYU_{fine_tuning_filename}training_PA_RH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}development_PA_RH_curv_ROI.pt',
                            f'NYU_{fine_tuning_filename}test_PA_RH_curv_ROI.pt']

                else:
                    return [
                        f'NYU_{fine_tuning_filename}training_pRFsize_RH_curv_ROI.pt',
                        f'NYU_{fine_tuning_filename}development_pRFsize_RH_curv_ROI.pt',
                        f'NYU_{fine_tuning_filename}test_pRFsize_RH_curv_ROI.pt']

    # def download(self):
    #     raise RuntimeError(
    #         'Dataset not found. Please download S1200_7T_Retinotopy_9Zkk.zip '
    #         'from {} and '
    #         'move it to {} and execute SettingDataset.sh'.format(self.url,
    #                                                              self.raw_dir))

    def process(self):
        # extract_zip(self.raw_paths[0], self.raw_dir, log=False)
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

        for i in range(0, self.n_examples):
            data = read_NYU(path, Hemisphere=self.hemisphere, index=i,
                            surface='mid', visual_mask_L=final_mask_L,
                            visual_mask_R=final_mask_R, faces_L=faces_L,
                            faces_R=faces_R, myelination=self.myelination,
                            prediction=self.prediction)
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        if self.fine_tuning:
            # Generate a train set of size (num_train_subjects) and a test set
            train = data_list[0:self.num_train_subjects]
            test = data_list[self.num_train_subjects:len(data_list)]
            # Save sets
            torch.save(self.collate(train), self.processed_paths[0])
            torch.save(self.collate(test), self.processed_paths[2])
        else:
            # (No fine-tuning) Generate only a test set
            test = data_list[0:len(data_list)]
            # Save set
            torch.save(self.collate(test), self.processed_paths[2])
        
