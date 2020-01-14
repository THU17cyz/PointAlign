import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from pointnet2_modules import PointnetSAModule, PointnetFPModule, PointnetSAModuleMSG
import numpy as np

class PointNet2_SSN(nn.Module):
    """
    """
    def __init__(self, num_classes):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(     # 0
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.2],
                nsamples=[64],
                mlps=[[6, 64, 64, 128]],
                first_layer=True,
                use_xyz=True,
            )
        )
        self.SA_modules.append(    # 1
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.4],
                nsamples=[64],
                mlps=[[128+9, 128, 128, 256]],
                use_xyz=False,
                last_layer=True,
            )
        )
        self.SA_modules.append(   # 4   global pooling
            PointnetSAModule(
                nsample=128,
                mlp=[256, 256, 512, 1024],
                use_xyz=False
            )
        )
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            PointnetFPModule(mlp=[128, 128, 128, 128])
        )
        self.FP_modules.append(PointnetFPModule(mlp=[384, 256, 128]))
        self.FP_modules.append(
            PointnetFPModule(mlp=[1280, 256, 256])
        )

        self.FC_layer = nn.Sequential(
            pt_utils.Conv1d(128, 128, bn=True), nn.Dropout(),
            pt_utils.Conv1d(128, num_classes, activation=None)
        )

    def forward(self, pc, normal, cls):
        """
        pc shape: (B, N, 3 + input_channels)
        formatted as (x, y, z, features...)
        """
        xyz = pc[..., 0:3].contiguous()
        if pc.size(-1) > 3:
            features = pc[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None
        
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            if i < 2:
                li_xyz, normal, li_features = self.SA_modules[i](l_xyz[i], normal, l_features[i])
            else:
                li_xyz, normal, li_features = self.SA_modules[i](l_xyz[i], None, l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        for i in range(len(l_features)):
            if 2 > i > 0:
                l_features[i] = l_features[i][:, 3:, :]
        
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])

        return self.FC_layer(l_features[0]).transpose(1, 2).contiguous()
