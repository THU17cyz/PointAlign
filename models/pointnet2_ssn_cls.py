import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
import numpy as np


# PointNet++: Single-Scale Neighborhood
class PointNet2_SSN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.2],
                nsamples=[32],
                mlps=[[6, 64, 64, 128]],
                use_xyz=True,
                first_layer=True,
            )
        )

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.4],
                nsamples=[64],
                mlps=[[128 + 9, 128, 128, 256]],
                use_xyz=False,
                last_layer=True
            )
        )
        
        self.SA_modules.append(
            # global convolutional pooling
            PointnetSAModule(
                nsample=128,
                mlp=[256, 256, 512, 1024], 
                use_xyz=False
            )
        )

        self.FC_layer = nn.Sequential(
            pt_utils.FC(1024, 512, activation=nn.ReLU(inplace=True), bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(512, 256, activation=nn.ReLU(inplace=True), bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(256, num_classes, activation=None)
        )

    def forward(self, pc, normal):
        """
        pc shape: (B, N, 3 + input_channels)
        formatted as (x, y, z, features...)
        """
        xyz = pc[..., 0:3].contiguous()
        if pc.size(-1) > 3:
            features = pc[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None
        for module in self.SA_modules:
            xyz, normal, features = module(xyz, normal, features)

        return self.FC_layer(features.squeeze(-1))


if __name__ == "__main__":
    pass
