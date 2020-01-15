import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
import pytorch_utils as pt_utils
from rscnn_modules import RSCNNSAModule, RSCNNSAModuleMSG


class RSCNN_MSN(nn.Module):
    """
    Relation-Shape CNN: Multi-Scale Neighborhood
    Classification Task
    """
    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        
        self.SA_modules.append(
            RSCNNSAModuleMSG(
                npoint=512,
                radii=[0.15, 0.23],
                nsamples=[24, 48],
                mlps=[[input_channels, 128], [input_channels, 128]],
                first_layer=True,
                scale_num=2,
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 128*2

        c_in = c_out_0
        self.SA_modules.append(
            RSCNNSAModuleMSG(
                npoint=256,
                radii=[0.2, 0.32],
                nsamples=[32, 64],
                mlps=[[c_in, 256], [c_in, 256]],
                scale_num=2,
                use_xyz=False,
            )
        )
        c_in = 512
        self.SA_modules.append(
            RSCNNSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.32],
                nsamples=[32, 64],
                mlps=[[c_in, 512], [c_in, 512]],
                scale_num=2,
                use_xyz=False,
                last_layer=True,
            )
        )
        
        self.SA_modules.append(
            # global pooling
            RSCNNSAModule(
                nsample=128,
                mlp=[1024, 1024],
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

    def forward(self, pc: torch.cuda.FloatTensor, normal: torch.cuda.FloatTensor):
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