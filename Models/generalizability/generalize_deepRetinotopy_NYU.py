import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import sys
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv

from Retinotopy.dataset.NYU_3sets_ROI import Retinotopy

"""
Used to test models trained on HCP training data (with a standard
processing pipeline applied), using only curvature in the feature set.
The models will be tested using unseen data from the NYU retinotopic
dataset (in a Test set). The 5 different trained models will be loaded, 
and used to generate Test set predictions.

This code can also be used to finetune models pre-trained on the HCP data
(with standard processing), and test the finetuned models' performance on data
points from the NYU dataset. If finetuning is performed, the number of 
participants to be added to the set used for finetuning must be specified,
as well as the number of epochs for which finetuning will take place.

Note: code implementation assumes that the file is being run from the dir 
explainability_CorticalSurfaceGDL/Models/generalizability - I have modified 
the code to automatically set the working dir to this (if it isn't already).
"""
# Set the working directory to Models/generalizability
os.chdir(osp.dirname(osp.realpath(__file__)))

#### Params used for model predictions ####
# Which hemisphere will predictions be generated for? ('Left'/'Right')
hemisphere = 'Right'
# What retinotopic characteristic will be predicted? ('eccentricity'/'polarAngle')
prediction = 'polarAngle'
'''
How many participants will be allocated to a 'Training' set for finetuning?
If num_finetuning_subjects == None, finetuning will not be performed.
'''
num_finetuning_subjects = 8
'''
How many epochs will finetuning occur for? If num_finetuning_subjects == None,
the value of num_epochs is ignored (as finetuning won't take place).
'''
num_epochs = 20

# Create the file name components for the chosen prediction params
HEMI_FILENAME = f'{hemisphere[0]}H'
if prediction == 'polarAngle':
    PRED_FILENAME = 'PA'
else:
    # prediction == 'eccentricity':
    PRED_FILENAME = 'ECC'
# Add additional info to filenames if finetuning is being used
FT_FILENAME = ""
if num_finetuning_subjects is not None:
    # Add the number of subjects used to finetune and number of epochs
    FT_FILENAME = \
        f'_finetuned_{num_finetuning_subjects}subj_{num_epochs}epochs'


# The number of participants (total) in NYU dataset
N_EXAMPLES = 43

'''
Used in the transform applied to the data. Normalization of the transform is
performed based on this value, instead of using the maximum possible value
observed in the data.
'''
NORM_VALUE = 70.4237

# Configure filepaths
sys.path.append('..')
path = osp.join(osp.dirname(osp.realpath(__file__)), '../../Retinotopy',
                'data')

# A pre-transform to be applied to the data
pre_transform = T.Compose([T.FaceToEdge()])

# If performing finetuning, load a Train set of some NYU data points
if num_finetuning_subjects is not None:
    train_dataset = Retinotopy(path, 'Train',
                          transform=T.Cartesian(max_value=NORM_VALUE),
                          pre_transform=pre_transform, n_examples=N_EXAMPLES,
                          prediction=prediction, hemisphere=hemisphere,
                          num_finetuning_subjects=num_finetuning_subjects)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# Loading Test set of remaining NYU data points
test_dataset = Retinotopy(path, 'Test',
                          transform=T.Cartesian(max_value=NORM_VALUE),
                          pre_transform=pre_transform, n_examples=N_EXAMPLES,
                          prediction=prediction, hemisphere=hemisphere,
                          num_finetuning_subjects=num_finetuning_subjects)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Model
class Net(torch.nn.Module):
    def __init__(self):
        """
        Initialise the model. Create 12 convolutional layers, using a B-spline 
        kernel. Configure methods to perform batch normalisation for each 
        respective convolutional layer.
        """
        super(Net, self).__init__()

        '''
        Input layer
        The number of feature maps (input channels) in the first convolutional 
        layer is 1 if curvature is the only input variable in the feature set. 
        If myelination was used, 2 feature maps would be used in this layer.
        '''
        self.conv1 = SplineConv(1, 8, dim=3, kernel_size=25)
        self.bn1 = torch.nn.BatchNorm1d(8)

        self.conv2 = SplineConv(8, 16, dim=3, kernel_size=25)
        self.bn2 = torch.nn.BatchNorm1d(16)

        self.conv3 = SplineConv(16, 32, dim=3, kernel_size=25)
        self.bn3 = torch.nn.BatchNorm1d(32)

        self.conv4 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn4 = torch.nn.BatchNorm1d(32)

        self.conv5 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn5 = torch.nn.BatchNorm1d(32)

        self.conv6 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn6 = torch.nn.BatchNorm1d(32)

        self.conv7 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn7 = torch.nn.BatchNorm1d(32)

        self.conv8 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn8 = torch.nn.BatchNorm1d(32)

        self.conv9 = SplineConv(32, 32, dim=3, kernel_size=25)
        self.bn9 = torch.nn.BatchNorm1d(32)

        self.conv10 = SplineConv(32, 16, dim=3, kernel_size=25)
        self.bn10 = torch.nn.BatchNorm1d(16)

        self.conv11 = SplineConv(16, 8, dim=3, kernel_size=25)
        self.bn11 = torch.nn.BatchNorm1d(8)

        # Final conv layer - flatten to one output channel
        self.conv12 = SplineConv(8, 1, dim=3, kernel_size=25)

    def forward(self, data):
        """
        Performs a single forward step in the computation of the NN.
        For each convolutional layer:
        Input variables, edge indexes, and edge attributes are provided to the 
        layer, and the ELU (Exponential Linear Unit) activation function is 
        then applied to that layer.
        Batch normalisation is performed, then dropout (a random 10% of
        units in each layer are zeroed).

        Args:
            data (Torch DataLoader object): includes the input features, as
                                            well as information on the input
                                            mesh (eg. edge index and edge 
                                            attributes) used in each conv layer
        Returns:
            x (Torch tensor): the output layer of the NN (the computed result
                              from the network's initial inputs applied through
                              each layer)
        """
        x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, pseudo))
        x = self.bn1(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv2(x, edge_index, pseudo))
        x = self.bn2(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv3(x, edge_index, pseudo))
        x = self.bn3(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv4(x, edge_index, pseudo))
        x = self.bn4(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv5(x, edge_index, pseudo))
        x = self.bn5(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv6(x, edge_index, pseudo))
        x = self.bn6(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv7(x, edge_index, pseudo))
        x = self.bn7(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv8(x, edge_index, pseudo))
        x = self.bn8(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv9(x, edge_index, pseudo))
        x = self.bn9(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv10(x, edge_index, pseudo))
        x = self.bn10(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv11(x, edge_index, pseudo))
        x = self.bn11(x)
        x = F.dropout(x, p=.10, training=self.training)

        # Output layer - applying ELU activation function
        x = F.elu(self.conv12(x, edge_index, pseudo)).view(-1)
        return x

def train(epoch):
    """
    Finetune the already trained model for one epoch.
    If required, update the optimiser learning rate to decay from 0.01 to 0.005.
    Calculate the training loss and training error, then update the model
    parameters.

    Args:
        epoch (int): the epoch for which the model is currently being trained
    Returns:
        output loss and MAE (Torch tensors): the model's respective training 
                                             loss (smooth L1) and training
                                             error (MAE)
    """
    model.train()

    # Set the learning rate to decay from 0.01 to 0.005 on the 100th epoch
    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

    for data in train_loader:
        # Assign the data point to hardware resources
        data = data.to(device)
        # Zero out computation graph for each training loop
        optimizer.zero_grad()

        # Calculate training loss - smooth L1 loss
        loss = torch.nn.SmoothL1Loss()
        # Weight prediction and ground truth by confidence measure
        output_loss = loss(model(data), data.y.view(-1))
        # Backpropagation
        output_loss.backward()

        # Calculate training error - MAE (Mean Absolute Error)
        MAE = torch.mean(abs(
            data.to(device).y.view(-1) - model(data))).item()
            # To check the performance of the model while training

        '''
        Note: bias has been removed from the loss/error functions for the
        model finetuning process. This bias was used when training the models on
        the HCP dataset (both for data points with the HCP-specific processing
        pipeline, and for data points with a standard processing pipeline using
        only curvature data). This was used during model training to create 
        better early visual cortex predictions.
        '''
        # Update the model parameters
        optimizer.step()

    return output_loss.detach(), MAE


# Make predictions on unseen NYU data for all 5 trained models
for i in range(5):
    # Check if CUDA is available for training with GPU resources
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Transfer Net tensor to the assigned hardware resources
    model = Net().to(device)

    # Load the trained model
    model.load_state_dict(
        torch.load(f'./../output/deepRetinotopy_{PRED_FILENAME}_' +
        f'{HEMI_FILENAME}_model' + str(i + 1) + '.pt', map_location=device))

    # If finetuning, create and use Adam optimizer algorithm on model
    if num_finetuning_subjects is not None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # Output folder name for Test set results
        directory = './NYU_testset_finetuned_results'
    else:
        # Output folder name for Test set results (no finetuning)
        directory = './NYU_testset_results'
    # Create output folder
    if not osp.exists(directory):
        os.makedirs(directory)

    # Perform finetuning (if required) for the specified # of epochs
    if num_finetuning_subjects is not None:
        for epoch in range(1, num_epochs+1):
            loss, MAE = train(epoch)
            print(
                'Epoch: {:02d}, Train_loss: {:.4f}, Train_MAE: {:.4f}'.format(
                    epoch, loss, MAE))

    def test():
        """
        Test the model's performance on the Test set - evaluate 
        the model and calculate the validation error for the given set (Mean 
        Absolute Error).
        Returns:
            output (dict): includes the predicted target values (y hat), 
                           measured target values (y), and MAE.
        """
        '''
        .eval() - weights of models are fixed (not calculating derivatives)
        If eval() is not called, test method will be used to finetune the model 
        with additional data.
        '''
        if num_finetuning_subjects is None:
            # If not finetuning, set model to evaluation mode (fix model weights)
            model.eval()

        MeanAbsError = 0
        y = []
        y_hat = []

        # For Test set:
        for data in test_loader:
            # Get predicted target values (y hat)
            pred = model(data.to(device)).detach()
            y_hat.append(pred)
            # Get observed target values (y)
            y.append(data.to(device).y.view(-1))
            # Calculate test error per data point - MAE
            MAE = torch.mean(abs(data.to(device).y.view(-1) - pred)).item()
            MeanAbsError += MAE

        # Calculate test error for entire Dev/Test set - MAE
        test_MAE = MeanAbsError / len(test_loader)

        output = {'Predicted_values': y_hat, 'Measured_values': y,
                  'MAE': test_MAE}
        return output

    # Test the model
    evaluation = test()

    # Save the Test set predictions and measured values
    torch.save({'Predicted_values': evaluation['Predicted_values'],
            'Measured_values': evaluation['Measured_values']},
            osp.join(osp.dirname(osp.realpath(__file__)),
                    directory[2:], f'NYU_testset{FT_FILENAME}-intactData_' +
                        f'{PRED_FILENAME}_{HEMI_FILENAME}_model' + str(
                        i + 1) + '.pt'))
