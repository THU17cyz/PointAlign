import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm2d as BN
import pointnet2_utils
import pytorch_utils as pt_utils
from typing import List
import numpy as np
import time
import math


def MLP(channels, batch_norm=True):
    """
    return a MLP of shape 'channels'
    """
    mlps = []
    for i in range(1, len(channels)):
        mlp = nn.Conv2d(in_channels=channels[i-1], out_channels=channels[i], kernel_size=(1, 1), stride=(1, 1), bias=True)
        nn.init.kaiming_normal_(mlp.weight)
        mlps.append(mlp)
    if batch_norm:
        mlp = Seq(*[
            Seq(mlps[i - 1], BN(channels[i]), ReLU())
            for i in range(1, len(channels))
        ])
    else:
        mlp = Seq(*[
            Seq(mlps[i - 1], ReLU())
            for i in range(1, len(channels))
        ])
    return mlp


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz, normal, features=None):
        """
        Parameters
        ----------
        xyz : (B, N, 3) tensor of the xyz coordinates of the points
        xyz : (B, N, 3) tensor of the normal vectors of the points
        features : (B, N, C) tensor of the descriptors of the the points

        Returns
        -------
        new_xyz : (B, npoint, 3) tensor of the new points' xyz
        new_normal : (B, npoint, 3) tensor of the new points' normal
        new_features : (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_points descriptors
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        
        if self.npoint is not None:
            normal_flipped = normal.transpose(1, 2).contiguous()
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, fps_idx).transpose(1, 2).contiguous()
            new_normal = pointnet2_utils.gather_operation(normal_flipped, fps_idx).transpose(1, 2).contiguous()
            fps_idx = fps_idx.data
        else:
            # for global convolution
            new_xyz = torch.FloatTensor([0.0]).cuda().unsqueeze(-1).unsqueeze(-1).expand(xyz.shape[0], 1, 3).contiguous()
            new_normal = None
            fps_idx = None
        
        for i in range(len(self.groupers)):
            if self.npoint is not None:
                new_features = self.groupers[i](xyz, new_xyz, normal, features, fps_idx)
            else:
                new_features = self.groupers[i](xyz, new_xyz, features)
            new_features = self.mlps[i]((new_features, new_normal))
            new_features_list.append(new_features)
        return new_xyz, new_normal, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """
    Pointnet set abstraction layer with multiscale grouping
    Parameters
    ----------
    npoint : number of points
    radii : list of radii to group with
    nsamples : number of samples in each ball query
    mlps : mlps for each scale
    use_xyz : use xyz or not
    first_layer: if it is the first layer
    last_layer: if it is the last layer
    """
    def __init__(
            self,
            *,
            npoint: int,
            radii: List[float],
            nsamples: List[int],
            mlps: List[List[int]],
            use_xyz: bool = True,
            bias=True,
            init=nn.init.kaiming_normal,
            first_layer=False,
            last_layer=False,
    ):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        mlp = MLP(mlps[0])
        
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if npoint is not None:
                self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
                self.mlps.append(pt_utils.PointNetConv(mlp=mlp, first_layer=first_layer, last_layer=last_layer))
            else:
                # global pooling
                self.groupers.append(pointnet2_utils.GroupAll(False))
                self.mlps.append(pt_utils.GloAvgConv(mlp=MLP(mlps[i])))


class PointnetSAModule(PointnetSAModuleMSG):
    """
    Parameters
    ----------
    mlp : mlps for each scale
    npoint : number of features
    radius : radius of ball
    nsample : number of samples in the ball query
    """
    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            use_xyz: bool = True,
    ):
        super().__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            use_xyz=use_xyz
        )


class PointnetFPModule(nn.Module):
    """
    Propagates the features of one set to another
    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor):
        """
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        dist, idx = pointnet2_utils.three_nn(unknown, known)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(
            known_feats, idx, weight
        )
        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats],
                                     dim=1)  #(B, C2 + C1, n)
        else:
            new_features = interpolated_feats
        
        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    pass
