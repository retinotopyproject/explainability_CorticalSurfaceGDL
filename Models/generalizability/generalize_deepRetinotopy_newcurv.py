import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import sys

sys.path.append('..')

from Retinotopy.dataset.HCP_3sets_ROI import Retinotopy
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../Retinotopy',
                'data')
pre_transform = T.Compose([T.FaceToEdge()])
norm_value = 70.4237

# What hemisphere will sets be generated for? (set to 'Left' or 'Right')
hemisphere = 'Left'
# What predicted retinotopic feature? (set to 'polarAngle' or 'eccentricity')
prediction = 'polarAngle'

# For formatting of loaded/saved filename strings
hemi_filename = hemisphere[0] # 'L' or 'R'
if prediction == 'polarAngle':
    prediction_filename = 'PA'
else:
    # prediction == 'eccentricity'
    prediction_filename = 'ECC'

# Loading dev and test dataset (10 subjects in each)
# Myelination data is ignored (using curvature data only)
dev_dataset = Retinotopy(path, 'Development',
                         transform=T.Cartesian(max_value=norm_value),
                         pre_transform=pre_transform, n_examples=181,
                         prediction=prediction, myelination=False,
                         hemisphere=hemisphere)
dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
test_dataset = Retinotopy(path, 'Test',
                          transform=T.Cartesian(max_value=norm_value),
                          pre_transform=pre_transform, n_examples=181,
                          prediction=prediction, myelination=False,
                          hemisphere=hemisphere)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        '''
        No. of feature maps is 1 if only using curvature data 
        (2 feature maps if myelination=True)
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

        self.conv12 = SplineConv(8, 1, dim=3, kernel_size=25)

    def forward(self, data):
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

        x = F.elu(self.conv12(x, edge_index, pseudo)).view(-1)
        return x

# Loading all 5 training models
for i in range(5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    model.load_state_dict(
        torch.load(f'./../output/deepRetinotopy_{prediction_filename}_' +
        f'{hemi_filename}H_model' + str(i + 1) + '.pt', map_location=device))

    # Create an output folder for dev set if it doesn't already exist
    directory = './devset_results'
    if not osp.exists(directory):
        os.makedirs(directory)

    # Creating output folder for test set results
    directory = './testset_results'
    if not osp.exists(directory):
        os.makedirs(directory)


    def test():
        model.eval()
        '''
        .eval() - weights of models are fixed (not calculating derivatives).
        If eval() is removed, can use to fine-tune model with additional data.
        '''
        MeanAbsError = 0
        y = []
        y_hat = []
        # For dev set:
        for data in dev_loader:
            pred = model(data.to(device)).detach()
            y_hat.append(pred)
            y.append(data.to(device).y.view(-1))
            MAE = torch.mean(abs(data.to(device).y.view(-1) - pred)).item()
            MeanAbsError += MAE
        test_MAE = MeanAbsError / len(dev_loader)
        
        # For test set:
        for data in test_loader:
            pred = model(data.to(device)).detach()
            y_hat.append(pred)
            y.append(data.to(device).y.view(-1))
            MAE = torch.mean(abs(data.to(device).y.view(-1) - pred)).item()
            MeanAbsError += MAE
        test_MAE = MeanAbsError / len(test_loader)

        output = {'Predicted_values': y_hat, 'Measured_values': y,
                  'MAE': test_MAE}
        return output


    evaluation = test()

    # Save the dev set
    torch.save({'Predicted_values': evaluation['Predicted_values'],
                'Measured_values': evaluation['Measured_values']},
               osp.join(osp.dirname(osp.realpath(__file__)),
                        'devset_results', f'devset-intactData_' +
                        f'{prediction_filename}_{hemi_filename}H_model' + str(
                        i + 1) + '.pt'))

    # Save the test set
    torch.save({'Predicted_values': evaluation['Predicted_values'],
            'Measured_values': evaluation['Measured_values']},
            osp.join(osp.dirname(osp.realpath(__file__)),
                    'testset_results', f'testset-intactData_' +
                        f'{prediction_filename}_{hemi_filename}H_model' + str(
                        i + 1) + '.pt'))
