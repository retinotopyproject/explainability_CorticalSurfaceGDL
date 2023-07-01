import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import sys
import time
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv

from Retinotopy.dataset.HCP_stdprocessing_3sets_ROI import Retinotopy

"""
Used to create and train models on HCP training data (with a standard
processing pipeline applied), using only curvature in the feature set. 
5 different models will be trained, with their training and development set 
performance evaluated. The performance of the development sets will be compared
later for hyperparameter tuning.
"""

#### Params used for model predictions ####
# Which hemisphere will predictions be generated for? ('Left'/'Right')
hemisphere = 'Right'
# What retinotopic characteristic will be predicted? ('eccentricity'/'polarAngle')
prediction = 'polarAngle'

# Create the file name components for the chosen prediction params
HEMI_FILENAME = f'{hemisphere[0]}H'
if prediction == 'polarAngle':
    PRED_FILENAME = 'PA'
else:
    # prediction == 'eccentricity':
    PRED_FILENAME = 'ECC'


# The number of participants (total) in all model sets
N_EXAMPLES = 181

'''
Used in the transform applied to the data. Normalization of the transform is
performed based on this value, instead of using the maximum possible value
observed in the data.
'''
NORM_VALUE = 70.4237

# Configure filepaths for saving the trained models
sys.path.append('..')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'Retinotopy', 'data')

# A pre-transform to be applied to the data
pre_transform = T.Compose([T.FaceToEdge()])

# Create Training set (for training the model)
train_dataset = Retinotopy(path, 'Train', 
                           transform=T.Cartesian(max_value=NORM_VALUE),
                           pre_transform=pre_transform, n_examples=N_EXAMPLES,
                           prediction=prediction, myelination=False,
                           hemisphere=hemisphere)

# Create Development dataset (hyperparameter tuning of the model)
dev_dataset = Retinotopy(path, 'Development', 
                         transform=T.Cartesian(max_value=NORM_VALUE),
                         pre_transform=pre_transform, n_examples=N_EXAMPLES,
                         prediction=prediction, myelination=False,
                         hemisphere=hemisphere)
                         
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)


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
        If myelination == True, 2 feature maps would be used in this layer.
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

# Check if CUDA is available for training with GPU resources
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Transfer Net tensor to the assigned hardware resources
model = Net().to(device)

# Use Adam optimizer algorithm on model, with initial learning rate gamma=0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    """
    Train the model for one epoch.
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
        # Set gradients of all optimised tensors to 0
        optimizer.zero_grad()

        # Get R2 values (and R2 values over a threshold of 2.2)
        R2 = data.R2.view(-1)
        threshold = R2.view(-1) > 2.2

        # Calculate training loss - smooth L1 loss
        loss = torch.nn.SmoothL1Loss()
        output_loss = loss(R2 * model(data), R2 * data.y.view(-1))
        output_loss.backward()

        # Calculate training error - MAE (Mean Absolute Error)
        MAE = torch.mean(abs(
            data.to(device).y.view(-1)[threshold == 1] - model(data)[
                threshold == 1])).item()  # To check the performance of the
        # model while training

        # Update the model parameters
        optimizer.step()

    return output_loss.detach(), MAE


def test():
    """
    Test the model's performance on the Development set - evaluate the model
    and calculate the test error (Mean Absolute Error).
    Returns:
        output (dict): includes the predicted target values (y hat), measured 
        target values (y), R2, MAE, and MAE for target values exceeding an R2
        threshold of 17.
    """
    # Set model to evaluation mode (model weights are fixed)
    model.eval()

    MeanAbsError = 0
    MeanAbsError_thr = 0
    y = []
    y_hat = []
    R2_plot = []

    for data in dev_loader:
        # Get predicted target values (y hat)
        pred = model(data.to(device)).detach()
        y_hat.append(pred)
        # Get observed target values (y)
        y.append(data.to(device).y.view(-1))

        # Get R2 values (and R2 values over certain thresholds)
        R2 = data.R2.view(-1)
        R2_plot.append(R2)
        threshold = R2.view(-1) > 2.2
        threshold2 = R2.view(-1) > 17

        # Calculate test error per data point - MAE (two different R2 thresholds)
        MAE = torch.mean(abs(data.to(device).y.view(-1)[threshold == 1] - pred[
            threshold == 1])).item()  # To check the performance of the
        # model
        MAE_thr = torch.mean(abs(
            data.to(device).y.view(-1)[threshold2 == 1] - pred[
                threshold2 == 1])).item()  # To check the performance of the
        # model
        MeanAbsError_thr += MAE_thr
        MeanAbsError += MAE

    # Calculate test error for whole Development set - MAE
    test_MAE = MeanAbsError / len(dev_loader)
    test_MAE_thr = MeanAbsError_thr / len(dev_loader)
    output = {'Predicted_values': y_hat, 'Measured_values': y, 'R2': R2_plot,
              'MAE': test_MAE, 'MAE_thr': test_MAE_thr}
    return output


# To find out how long it takes to train the model:
# init = time.time() 

# Create an output folder if it doesn't already exist
directory = './output'
if not osp.exists(directory):
    os.makedirs(directory)
    
# Train 5 distinct models
for i in range(5):
    # Train each model (and evaluate dev set performance) for 200 epochs
    for epoch in range(1, 201):
        loss, MAE = train(epoch)
        test_output = test()
        # Display the train and dev set metrics for each epoch
        print(
            'Epoch: {:02d}, Train_loss: {:.4f}, Train_MAE: {:.4f}, Test_MAE: '
            '{:.4f}, Test_MAE_thr: {:.4f}'.format(
                epoch, loss, MAE, test_output['MAE'], test_output['MAE_thr']))

    # Saving model's learned parameters
    torch.save(model.state_dict(),
               osp.join(osp.dirname(osp.realpath(__file__)), 'output',
                    f'deepRetinotopy_{PRED_FILENAME}_{HEMI_FILENAME}_model' + 
                    str(i+1) + '.pt'))

# To find out how long it takes to train the model:
# end = time.time() 
# time = (end - init) / 60
# print(str(time) + ' minutes')
