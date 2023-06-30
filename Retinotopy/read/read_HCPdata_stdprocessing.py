import scipy.io
import numpy as np
import torch
import os.path as osp
from numpy.random import seed
from torch_geometric.data import Data
import nibabel as nib


def read_HCP(path, Hemisphere=None, index=None, surface=None, threshold=None,
             shuffle=True, visual_mask_L=None, visual_mask_R=None,
             faces_L=None, faces_R=None, myelination=None, prediction=None):
    """
    Reads participant data from the HCP dataset, with a standard processing
    pipeline applied.
    Read the data files and create a data object with attributes x, y, pos,
    faces and R2.

        Args:
            path (string): Path to raw dataset
            Hemisphere (string): 'Left' or 'Right' hemisphere
            index (int): Index of the participant
            surface (string): Surface template ('mid'). If surface=='sphere',
                              the method will throw an exception.
            threshold (float): threshold for selection of vertices in the
                ROI based on the R2 of pRF modelling
            shuffle (boolean): shuffle the participants' IDs list
            visual_mask_L (numpy array): Mask of the region of interest from
                left hemisphere (32492,)
            visual_mask_R (numpy array): Mask of the region of interest from
                right hemisphere (32492,)
            faces_L (numpy array): triangular faces from the region of
                interest (number of faces, 3) in the left hemisphere
            faces_R (numpy array): triangular faces from the region of
                interest (number of faces, 3) in the right hemisphere
            myelination (boolean): True if myelin values will be used as an
                additional feature
            prediction (string): output of the model ('polarAngle' or
                'eccentricity')

        Returns:
            data (object): object of class Data (from torch_geometric.data)
                with attributes x, y, pos, faces and R2.
    """
    # Total number of cortical nodes in the mesh
    NUMBER_CORTICAL_NODES = int(64984)
    # Number of nodes within each hemisphere
    NUMBER_HEMI_NODES = int(NUMBER_CORTICAL_NODES / 2)

    '''
    The curvature data files (with standard processing applied) are loaded 
    independently for each hemisphere. Other features (ecc, PA, pRF size, R2,
    myelin) are loaded for both hemispheres.
    '''
    eccentricity = \
        scipy.io.loadmat(osp.join(path, 'cifti_eccentricity_all.mat'))[
            'cifti_eccentricity']
    polarAngle = scipy.io.loadmat(osp.join(path, 'cifti_polarAngle_all.mat'))[
        'cifti_polarAngle']
    pRFsize = scipy.io.loadmat(osp.join(path, 'cifti_pRFsize_all.mat'))[
        'cifti_pRFsize']
    R2 = scipy.io.loadmat(osp.join(path, 'cifti_R2_all.mat'))['cifti_R2']
    myelin = scipy.io.loadmat(osp.join(path, 'cifti_myelin_all.mat'))[
        'cifti_myelin']

    # Loading list of subjects
    with open(osp.join(path, '..', '..', 'list_subj')) as fp:
        subjects = fp.read().split("\n")
    subjects = subjects[0:len(subjects) - 1]

    # Shuffling the subjects - set the seed value for reproducible results
    seed(1)
    if shuffle == True:
        np.random.shuffle(subjects)

    '''
    Saving each participants' ID to a text file (in the order that participants
    are added to the training, dev, and test sets).
    For generating plots based on train/dev/test sets, to match the participants
    to the correct corresponding curvature maps. As data is being shuffled,
    the order of participants must be noted to generate accurate graphs.
    This output shouldn't be used for any purpose within the actual training
    or validation of the model.
    '''
    # f = open(osp.join(path, '..', '..', '..', 'participant_IDs_in_order.txt'), 
    #         "a")
    # f.write(f'{subjects[index]}\n')
    # print(f'Participant {index+1}: {subjects[index]}')
    # f.close()

    '''
    Note: the Right hemisphere contains nodes with indices from 
    NUMBER_HEMI_NODES up to NUMBER_CORTICAL_NODES - 1.
    '''
    if Hemisphere == 'Right':
        # Reading curvature data with standard processing
        curv_R = nib.load(osp.join(path, 'fs-curvature', f'{subjects[index]}/', 
        subjects[index] + '.R.curvature.32k_fs_LR.shape.gii'))
        curvature = torch.tensor(np.reshape(curv_R.agg_data()
                        .reshape((NUMBER_HEMI_NODES))
        [visual_mask_R == 1], (-1, 1)), dtype=torch.float)

        # Loading connectivity of triangles (faces of the mesh)
        faces = torch.tensor(faces_R.T, dtype=torch.long)  # Transforming data
        # to torch data type

        # Get coordinates of the Right hemisphere vertices
        if surface == 'mid':
            pos = torch.tensor((scipy.io.loadmat(
                osp.join(path, 'mid_pos_R.mat'))['mid_pos_R'].reshape(
                (NUMBER_HEMI_NODES, 3))[visual_mask_R == 1]),
                               dtype=torch.float)

        '''
        Reading curvature data (standard processing) for a spherical surface
        isn't configured. If surface == 'sphere', the method will throw an
        exception. 
        '''
        if surface == 'sphere':
            raise Exception("Reading HCP (standard proecessing) curvature " +
                "data with a spherical surface is not configured. Please set " +
                "the surface kwarg to 'mid' when calling this method.")

        # Loading measures for the Right hemisphere
        R2_values = torch.tensor(np.reshape(
            R2['x' + subjects[index] + '_fit1_r2_msmall'][0][0][
            NUMBER_HEMI_NODES:NUMBER_CORTICAL_NODES].reshape(
                (NUMBER_HEMI_NODES))[visual_mask_R == 1], (-1, 1)),
            dtype=torch.float)
        myelin_values = torch.tensor(np.reshape(
            myelin['x' + subjects[index] + '_myelinmap'][0][0][
            NUMBER_HEMI_NODES:NUMBER_CORTICAL_NODES].reshape(
                (NUMBER_HEMI_NODES))[visual_mask_R == 1], (-1, 1)),
            dtype=torch.float)
        eccentricity_values = torch.tensor(np.reshape(
            eccentricity['x' + subjects[index] + '_fit1_eccentricity_msmall'][
                0][0][
            NUMBER_HEMI_NODES:NUMBER_CORTICAL_NODES].reshape(
                (NUMBER_HEMI_NODES))[visual_mask_R == 1], (-1, 1)),
            dtype=torch.float)
        polarAngle_values = torch.tensor(np.reshape(
            polarAngle['x' + subjects[index] + '_fit1_polarangle_msmall'][0][
                0][
            NUMBER_HEMI_NODES:NUMBER_CORTICAL_NODES].reshape(
                (NUMBER_HEMI_NODES))[visual_mask_R == 1], (-1, 1)),
            dtype=torch.float)
        pRFsize_values = torch.tensor(np.reshape(
            pRFsize['x' + subjects[index] + '_fit1_receptivefieldsize_msmall'][
                0][0][
            NUMBER_HEMI_NODES:NUMBER_CORTICAL_NODES].reshape(
                (NUMBER_HEMI_NODES))[visual_mask_R == 1], (-1, 1)),
            dtype=torch.float)

        # Remove NaN values from curvature and myelination data
        nocurv = np.isnan(curvature)
        curvature[nocurv == 1] = 0

        nomyelin = np.isnan(myelin_values)
        myelin_values[nomyelin == 1] = 0

        # Remove NaNs from other measures (R2, ecc, PA, pRF size)
        noR2 = np.isnan(R2_values)
        R2_values[noR2 == 1] = 0

        # condition=R2_values < threshold
        condition2 = np.isnan(eccentricity_values)
        condition3 = np.isnan(polarAngle_values)
        condition4 = np.isnan(pRFsize_values)

        # eccentricity_values[condition == 1] = -1
        eccentricity_values[condition2 == 1] = -1

        # polarAngle_values[condition==1] = -1
        polarAngle_values[condition3 == 1] = -1

        pRFsize_values[condition4 == 1] = -1

        # Create a graph (data) containing the required features
        if myelination == False:
            if prediction == 'polarAngle':
                data = Data(x=curvature, y=polarAngle_values, pos=pos)
            elif prediction == 'eccentricity':
                data = Data(x=curvature, y=eccentricity_values, pos=pos)
            else:
                # if prediction == 'pRFsize':
                data = Data(x=curvature, y=pRFsize_values, pos=pos)
        else:
            if prediction == 'polarAngle':
                data = Data(x=torch.cat((curvature, myelin_values), 1),
                            y=polarAngle_values, pos=pos)
            elif prediction == 'eccentricity':
                data = Data(x=torch.cat((curvature, myelin_values), 1),
                            y=eccentricity_values, pos=pos)
            else:
                # if prediction == 'pRFsize':
                data = Data(x=torch.cat((curvature, myelin_values), 1),
                            y=pRFsize_values, pos=pos)
        # Store faces and R2 values in the graph (data) object
        data.face = faces
        data.R2 = R2_values

    '''
    Note: the Left hemisphere is made up of nodes with indices from the first
    node (at index 0) up to NUMBER_HEMI_NODES - 1.
    '''
    if Hemisphere == 'Left':
        # Reading curvature data with standard processing
        curv_L = nib.load(osp.join(path, 'fs-curvature', f'{subjects[index]}', 
        subjects[index] + '.L.curvature.32k_fs_LR.shape.gii'))
        curvature = torch.tensor(np.reshape(curv_L.agg_data()
                        .reshape((NUMBER_HEMI_NODES))
        [visual_mask_L == 1], (-1, 1)), dtype=torch.float)

        # Loading connectivity of triangles (faces of the mesh)
        faces = torch.tensor(faces_L.T, dtype=torch.long)  # Transforming data
        # to torch data type

        # Get coordinates of the Left hemisphere vertices
        if surface == 'mid':
            pos = torch.tensor((scipy.io.loadmat(
                osp.join(path, 'mid_pos_L.mat'))['mid_pos_L'].reshape(
                (NUMBER_HEMI_NODES, 3))[visual_mask_L == 1]),
                               dtype=torch.float)

        '''
        Reading curvature data (standard processing) for a spherical surface
        isn't configured. If surface == 'sphere', the method will throw an
        exception. 
        '''
        if surface == 'sphere':
            raise Exception("Reading HCP (standard proecessing) curvature " +
                "data with a spherical surface is not configured. Please set " +
                "the surface kwarg to 'mid' when calling this method.")

        # Loading measures for the Left hemisphere
        R2_values = torch.tensor(np.reshape(
            R2['x' + subjects[index] + '_fit1_r2_msmall'][0][0][
            0:NUMBER_HEMI_NODES].reshape((NUMBER_HEMI_NODES))[
                visual_mask_L == 1], (-1, 1)), dtype=torch.float)
        myelin_values = torch.tensor(np.reshape(
            myelin['x' + subjects[index] + '_myelinmap'][0][0][
            0:NUMBER_HEMI_NODES].reshape(
                (NUMBER_HEMI_NODES))[visual_mask_L == 1], (-1, 1)),
            dtype=torch.float)
        eccentricity_values = torch.tensor(np.reshape(
            eccentricity['x' + subjects[index] + '_fit1_eccentricity_msmall'][
                0][0][0:NUMBER_HEMI_NODES].reshape((NUMBER_HEMI_NODES))[
                visual_mask_L == 1], (-1, 1)), dtype=torch.float)
        polarAngle_values = torch.tensor(np.reshape(
            polarAngle['x' + subjects[index] + '_fit1_polarangle_msmall'][0][
                0][0:NUMBER_HEMI_NODES].reshape((NUMBER_HEMI_NODES))[
                visual_mask_L == 1], (-1, 1)), dtype=torch.float)
        pRFsize_values = torch.tensor(np.reshape(
            pRFsize['x' + subjects[index] + '_fit1_receptivefieldsize_msmall'][
                0][0][0:NUMBER_HEMI_NODES].reshape(
                (NUMBER_HEMI_NODES))[visual_mask_L == 1], (-1, 1)),
            dtype=torch.float)

        # Remove NaN values from curvature and myelination data
        nocurv = np.isnan(curvature)
        curvature[nocurv == 1] = 0

        nomyelin = np.isnan(myelin_values)
        myelin_values[nomyelin == 1] = 0

        # Remove NaNs from other measures (R2, ecc, PA, pRF size)
        noR2 = np.isnan(R2_values)
        R2_values[noR2 == 1] = 0

        # condition=R2_values < threshold
        condition2 = np.isnan(eccentricity_values)
        condition3 = np.isnan(polarAngle_values)
        condition4 = np.isnan(pRFsize_values)

        # eccentricity_values[condition == 1] = -1
        eccentricity_values[condition2 == 1] = -1

        # polarAngle_values[condition==1] = -1
        polarAngle_values[condition3 == 1] = -1

        pRFsize_values[condition4 == 1] = -1

        # Translating polar angle values
        sum = polarAngle_values < 180
        minus = polarAngle_values > 180
        polarAngle_values[sum] = polarAngle_values[sum] + 180
        polarAngle_values[minus] = polarAngle_values[minus] - 180

        # Create a graph (data) containing the required features
        if myelination == False:
            if prediction == 'polarAngle':
                data = Data(x=curvature, y=polarAngle_values, pos=pos)
            elif prediction == 'eccentricity':
                data = Data(x=curvature, y=eccentricity_values, pos=pos)
            else:
                # if prediction == 'pRFsize':
                data = Data(x=curvature, y=pRFsize_values, pos=pos)
        else:
            if prediction == 'polarAngle':
                data = Data(x=torch.cat((curvature, myelin_values), 1),
                            y=polarAngle_values, pos=pos)
            elif prediction == 'eccentricity':
                data = Data(x=torch.cat((curvature, myelin_values), 1),
                            y=eccentricity_values, pos=pos)
            else:
                # if prediction == 'pRFsize':
                data = Data(x=torch.cat((curvature, myelin_values), 1),
                            y=pRFsize_values, pos=pos)
        # Store faces and R2 values in the graph (data) object
        data.face = faces
        data.R2 = R2_values

    return data

