import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import sys

sys.path.append('..')

from Retinotopy.dataset.NYU_3sets_ROI import Retinotopy
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../Retinotopy',
                'data')
pre_transform = T.Compose([T.FaceToEdge()])
hemisphere = 'Left'  # or 'Right'
norm_value = 70.4237

# How many subjects will be allocated to a 'training' set for fine-tuning?
num_finetuning_subjects = 8
# How many epochs will fine-tuning occur for?
num_epochs = 10

# Loading test dataset
# Loading training set for fine tuning
train_dataset = Retinotopy(path, 'Train',
                          transform=T.Cartesian(max_value=norm_value),
                          pre_transform=pre_transform, n_examples=43,
                          prediction='polarAngle',
                          hemisphere=hemisphere,
                          num_finetuning_subjects=num_finetuning_subjects)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
# Loading test set (contains the remaining subjects in the dataset)
test_dataset = Retinotopy(path, 'Test',
                          transform=T.Cartesian(max_value=norm_value),
                          pre_transform=pre_transform, n_examples=43,
                          prediction='polarAngle',
                          hemisphere=hemisphere,
                          num_finetuning_subjects=num_finetuning_subjects)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        '''
        No. of feature maps is 1 if only using curvature data
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

def train(epoch):
    model.train()

    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

    for data in train_loader:
        data = data.to(device)
        # Zero out computation graph for each training loop
        optimizer.zero_grad()

        # R2 = data.R2.view(-1)
        # threshold = R2.view(-1) > 2.2

        loss = torch.nn.SmoothL1Loss()
        # Weight prediction and ground truth by confidence measure
        output_loss = loss(model(data), data.y.view(-1))
        # Backpropagation
        output_loss.backward()

        MAE = torch.mean(abs(
            data.to(device).y.view(-1) - model(data))).item()
            # To check the performance of the
        # model while training
        '''
        Bias removed (bias used to create better early visual cortex predictions)
        '''

        optimizer.step()
    return output_loss.detach(), MAE

def test():
    # model.eval()
    '''
    .eval() - weights of models are fixed (not calculating derivatives)
    If eval() is removed, can use to fine tune with additional data
    '''
    MeanAbsError = 0
    y = []
    y_hat = []
    # For dev set:
    # for data in dev_loader:
    #     pred = model(data.to(device)).detach()
    #     y_hat.append(pred)
    #     y.append(data.to(device).y.view(-1))
    #     MAE = torch.mean(abs(data.to(device).y.view(-1) - pred)).item()
    #     MeanAbsError += MAE
    # test_MAE = MeanAbsError / len(dev_loader)
    
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

for i in range(5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    model.load_state_dict(
        torch.load(
            './../output/deepRetinotopy_PA_LH_model' + str(i + 1) + '.pt',
            map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Create an output folder for dev set if it doesn't already exist
    # directory = './NYU_devset_fineTuned_results'
    # if not osp.exists(directory):
    #     os.makedirs(directory)

    # Creating output folder for test set results
    directory = './NYU_testset_fineTuned_results'
    if not osp.exists(directory):
        os.makedirs(directory)

    for epoch in range(1, num_epochs+1):
        loss, MAE = train(epoch)
        # test_output = test()
        # print(
        #     'Epoch: {:02d}, Train_loss: {:.4f}, Train_MAE: {:.4f}, Test_MAE: '
        #     '{:.4f}, Test_MAE_thr: {:.4f}'.format(
        #         epoch, loss, MAE, test_output['MAE'], test_output['MAE_thr']))
        print(
            'Epoch: {:02d}, Train_loss: {:.4f}, Train_MAE: {:.4f}'.format(
                epoch, loss, MAE))
        
    evaluation = test()

    # For dev set:
    # torch.save({'Predicted_values': evaluation['Predicted_values'],
    #             'Measured_values': evaluation['Measured_values']},
    #            osp.join(osp.dirname(osp.realpath(__file__)),
    #                     'NYU_devset_fineTuned_results',
    #                     'NYU_devset_fineTuned-intactData_PA_LH_model' + str(
    #                         i + 1) + '.pt'))

    # For test set:
    torch.save({'Predicted_values': evaluation['Predicted_values'],
            'Measured_values': evaluation['Measured_values']},
            osp.join(osp.dirname(osp.realpath(__file__)),
                    'NYU_testset_fineTuned_results',
                    f'NYU_testset_fineTuned_{num_finetuning_subjects}subj_{num_epochs}epochs-intactData_PA_LH_model' + str(
                        i + 1) + '.pt'))
