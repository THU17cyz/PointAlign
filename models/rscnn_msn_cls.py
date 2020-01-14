import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from rscnn_modules import RSCNNSAModule, RSCNNSAModuleMSG
import numpy as np

# Relation-Shape CNN: Single-Scale Neighborhood
class RSCNN_MSN(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        
        self.SA_modules.append(
            RSCNNSAModuleMSG(
                npoint=512,
                # radii = [0.23],
                # nsamples=[48],
                # mlps=[[input_channels, 128]],

                radii=[0.15, 0.23],
                nsamples=[24, 48],
                mlps=[[input_channels, 128], [input_channels, 128]],

                # radii=[0.1, 0.15, 0.23],
                # nsamples=[16, 24, 48],
                # mlps=[[input_channels, 128], [input_channels, 128], [input_channels, 128]],

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
                # radii = [0.32],
                # nsamples=[64],
                # mlps=[[128, 512]],

                radii=[0.2, 0.32],
                nsamples=[32, 64],
                mlps=[[c_in, 256], [c_in, 256]],

                # radii=[0.14, 0.2, 0.32],
                # nsamples=[16, 32, 64],
                # mlps=[[c_in, 256], [c_in, 256], [c_in, 256]],
                scale_num=2,
                use_xyz=False,
            )
        )
        c_in = 512
        self.SA_modules.append(
            RSCNNSAModuleMSG(
                npoint=128,
                # radii = [0.32],
                # nsamples=[64],
                # mlps=[[128, 512]],

                radii=[0.2, 0.32],
                nsamples=[32, 64],
                mlps=[[c_in, 512], [c_in, 512]],

                # radii=[0.14, 0.2, 0.32],
                # nsamples=[16, 32, 64],
                # mlps=[[c_in, 256], [c_in, 256], [c_in, 256]],
                scale_num=2,
                use_xyz=False,
                last_layer=True,
            )
        )
        
        self.SA_modules.append(
            # global convolutional pooling
            RSCNNSAModule(
                nsample = 128,
                mlp = [1024, 1024],
                #mlp=[512, 1024], 
                use_xyz=False #modified
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
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
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
    sim_data = Variable(torch.rand(32, 2048, 6))
    sim_data = sim_data.cuda()
    sim_cls = Variable(torch.ones(32, 16))
    sim_cls = sim_cls.cuda()

    seg = RSCNN_SSN(num_classes=50, input_channels=3, use_xyz=True)
    seg = seg.cuda()
    out = seg(sim_data, sim_cls)
    print('seg', out.size())