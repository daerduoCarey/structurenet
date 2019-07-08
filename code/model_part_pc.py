"""
    This file defines part point cloud VAE/AE model.
"""

import torch
import torch.nn as nn
from chamfer_distance import ChamferDistance
from collections import namedtuple


class PartFeatSampler(nn.Module):

    def __init__(self, feature_size, probabilistic=True):
        super(PartFeatSampler, self).__init__()
        self.probabilistic = probabilistic

        self.mlp2mu = nn.Linear(feature_size, feature_size)
        self.mlp2var = nn.Linear(feature_size, feature_size)

    def forward(self, x):
        mu = self.mlp2mu(x)

        if self.probabilistic:
            logvar = self.mlp2var(x)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)

            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu


class PartEncoder(nn.Module):

    def __init__(self, feat_len, probabilistic=False):
        super(PartEncoder, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, feat_len, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(feat_len)

        self.sampler = PartFeatSampler(feature_size=feat_len, probabilistic=probabilistic)

    def forward(self, pc):
        net = pc.transpose(2, 1)
        net = torch.relu(self.bn1(self.conv1(net)))
        net = torch.relu(self.bn2(self.conv2(net)))
        net = torch.relu(self.bn3(self.conv3(net)))
        net = torch.relu(self.bn4(self.conv4(net)))

        net = net.max(dim=2)[0]
        net = self.sampler(net)

        return net


class PartDecoder(nn.Module):

    def __init__(self, feat_len, num_point):
        super(PartDecoder, self).__init__()
        self.num_point = num_point

        self.mlp1 = nn.Linear(feat_len, feat_len)
        self.mlp2 = nn.Linear(feat_len, feat_len)
        self.mlp3 = nn.Linear(feat_len, num_point*3)

        self.bn1 = nn.BatchNorm1d(feat_len)
        self.bn2 = nn.BatchNorm1d(feat_len)

        self.chamferLoss = ChamferDistance()

    def forward(self, net):
        net = torch.relu(self.bn1(self.mlp1(net)))
        net = torch.relu(self.bn2(self.mlp2(net)))
        net = self.mlp3(net).view(-1, self.num_point, 3)

        return net

    def loss(self, pred, gt):
        dist1, dist2 = self.chamferLoss(pred, gt)
        loss = (dist1.mean(dim=1) + dist2.mean(dim=1)) / 2
        avg_loss = loss.mean() * 1000

        return avg_loss
