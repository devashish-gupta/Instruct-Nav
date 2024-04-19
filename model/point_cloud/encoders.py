import torch
import torch.nn as nn
from .pointnet_utils import *


class PointNetEncoder(nn.Module):
    '''
    Point cloud encoder based on PointNet.
    References:
        https://stanford.edu/~rqi/pointnet/
    '''
    def __init__(self):
        super().__init__()

        self.stn = STN3d()
        self.fstn = STNkd(k=64)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.emb_dim = 1024

    def set_mode(self, mode):
        if mode == 'train':
            self.train()
        else:
            self.eval()

    def forward(self, points, mode='train'):
        '''
        Forward method for the model

        Args:
            points (torch.Tensor): shape: (batch, 3, num_points)
            mode (str): 'train'/'eval'

        Returns:
            torch.Tensor: Global representation of the point cloud. shape: ()
        '''
        self.set_mode(mode)
        trans = self.stn(points) # input transformation
        x = points.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = self.fstn(x) # feature transformation
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x # returning global representation only