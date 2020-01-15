import torch
import torch.nn as nn
import torch.nn.functional as F

import pointnet2_utils
import pytorch_utils as pt_utils
from typing import List
import numpy as np
import time
import math


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz: torch.Tensor, normal: torch.Tensor,
                features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
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
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)  # (B, npoint)
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, fps_idx).transpose(1, 2).contiguous()
            new_normal = pointnet2_utils.gather_operation(normal_flipped, fps_idx).transpose(1, 2).contiguous()
            fps_idx = fps_idx.data
        else:
            new_xyz = None
            new_normal = None
            fps_idx = None
        
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, normal, features, fps_idx) if self.npoint is not None else self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            
            new_features = self.mlps[i](
                (new_features, new_normal)
            )  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)
        
        return new_xyz, new_normal, torch.cat(new_features_list, dim=1)


class RSCNNSAModuleMSG(_PointnetSAModuleBase):
    """
    RSCNN layer with multiscale grouping
    Parameters
    ----------
    npoint : number of points
    radii : list of radii to group with
    nsamples : number of samples in each ball query
    mlps : mlps for each scale
    use_xyz : use xyz or not
    first_layer: if it is the first layer
    last_layer: if it is the last layer
    scale_num: how many scales are used
    rel_pose_mode: how to calculate the relative pose, "first" means use the first, "avg" means averaging all
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
            scale_num=1,
            rel_pose_mode="first"
    ):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        
        # initialize shared mapping functions
        C_in = (mlps[0][0] + 3) if use_xyz else mlps[0][0]
        C_out = mlps[0][1]

        if first_layer:
            in_channels = 7
        else:
            in_channels = 10

        if first_layer:
            mapping_func1 = nn.Conv2d(in_channels=in_channels, out_channels=math.floor(C_out / 2), kernel_size=(1, 1),
                                      stride=(1, 1), bias=bias)
            mapping_func2 = nn.Conv2d(in_channels=math.floor(C_out / 2), out_channels=16, kernel_size=(1, 1),
                                  stride=(1, 1), bias=bias)
            xyz_raising = nn.Conv2d(in_channels=C_in, out_channels=16, kernel_size=(1, 1),
                                  stride=(1, 1), bias=bias)
            init(xyz_raising.weight)
            if bias:
                nn.init.constant(xyz_raising.bias, 0)
        elif npoint is not None:
            mapping_func1 = nn.Conv2d(in_channels=in_channels, out_channels=math.floor(C_out / 4), kernel_size=(1, 1),
                                      stride=(1, 1), bias=bias)
            mapping_func2 = nn.Conv2d(in_channels=math.floor(C_out / 4), out_channels=C_in, kernel_size=(1, 1),
                                  stride=(1, 1), bias=bias)

        if npoint is not None:
            init(mapping_func1.weight)
            init(mapping_func2.weight)
            if bias:
                nn.init.constant(mapping_func1.bias, 0)
                nn.init.constant(mapping_func2.bias, 0)    
                     
            # channel raising mapping
            cr_mapping = nn.Conv1d(in_channels=C_in if not first_layer else 16, out_channels=C_out, kernel_size=1,
                                      stride=1, bias=bias)
            init(cr_mapping.weight)
            nn.init.constant(cr_mapping.bias, 0)
        
        if first_layer:
            mapping = [mapping_func1, mapping_func2, cr_mapping, xyz_raising]
        elif npoint is not None:
            mapping = [mapping_func1, mapping_func2, cr_mapping]
        
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(False) # modified
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            if npoint is not None:
                self.mlps.append(pt_utils.SharedRSConv(mlp_spec, mapping = mapping, first_layer = first_layer, last_layer=last_layer, scale_num=scale_num, rel_pose_mode=rel_pose_mode))
            else:
                # global pooling
                self.mlps.append(pt_utils.RSCNNGloAvgConv(C_in = C_in, C_out = C_out))


class RSCNNSAModule(RSCNNSAModuleMSG):
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


class RSCNNFPModule(nn.Module):
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

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor,
            unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
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

        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    pass
