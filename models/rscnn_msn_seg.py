import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from rscnn_modules import RSCNNSAModule, RSCNNFPModule, RSCNNSAModuleMSG
import numpy as np

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

    def __init__(self, num_classes, input_channels=0, relation_prior=1, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(     # 0
            RSCNNSAModuleMSG(
                npoint=1024,
                radii=[0.075, 0.1, 0.125],
                nsamples=[16, 32, 48],
                mlps=[[c_in, 64], [c_in, 64], [c_in, 64]],
                first_layer=True,
                use_xyz=use_xyz,
                scale_num=3,
                rel_pose_mode="avg",
            )
        )
        c_out_0 = 64*3

        c_in = c_out_0
        self.SA_modules.append(    # 1
            RSCNNSAModuleMSG(
                npoint=256,
                radii=[0.1, 0.15, 0.2],
                nsamples=[16, 48, 64],
                mlps=[[c_in, 128], [c_in, 128], [c_in, 128]],
                use_xyz=False,
                scale_num=3,
                rel_pose_mode="avg",
            )
        )
        c_out_1 = 128*3

        c_in = c_out_1
        self.SA_modules.append(    # 2
            RSCNNSAModuleMSG(
                npoint=64,
                radii=[0.2, 0.3, 0.4],
                nsamples=[16, 32, 48],
                mlps=[[c_in, 256], [c_in, 256], [c_in, 256]],
                use_xyz=False,
                scale_num=3,
                rel_pose_mode="avg",
            )
        )
        c_out_2 = 256*3

        c_in = c_out_2
        self.SA_modules.append(    # 3
            RSCNNSAModuleMSG(
                npoint=16,
                radii=[0.4, 0.6, 0.8],
                nsamples=[16, 24, 32],
                mlps=[[c_in, 512], [c_in, 512], [c_in, 512]],
                use_xyz=False,
                last_layer=True,
                scale_num=3,
                rel_pose_mode="avg",
            )
        )
        c_out_3 = 512*3
        
        self.SA_modules.append(   # 4   global pooling
            RSCNNSAModule(
                nsample = 16,
                mlp=[c_out_3, 128], use_xyz=False
            )
        )
        global_out = 128
        
        self.SA_modules.append(   # 5   global pooling
            RSCNNSAModule(
                nsample = 64,
                mlp=[c_out_2, 128], use_xyz=False
            )
        )
        global_out2 = 128

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            RSCNNFPModule(mlp=[256 + input_channels, 128, 128])
        )
        self.FP_modules.append(RSCNNFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(RSCNNFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(
            RSCNNFPModule(mlp=[c_out_3 + c_out_2, 512, 512])
        )

        self.FC_layer = nn.Sequential(
            pt_utils.Conv1d(128+global_out+global_out2+16, 128, bn=True), nn.Dropout(),
            pt_utils.Conv1d(128, num_classes, activation=None)
        )

    def forward(self, pc: torch.cuda.FloatTensor, normal: torch.cuda.FloatTensor, cls):
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
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            if i < 5:
                if i < 4:
                    li_xyz, normal, li_features = self.SA_modules[i](l_xyz[i], normal, l_features[i])
                else:
                    
                    li_xyz, normal, li_features = self.SA_modules[i](l_xyz[i], None, l_features[i])
                if li_xyz is not None:
                    random_index = np.arange(li_xyz.size()[1])
                    np.random.shuffle(random_index)
                    li_xyz = li_xyz[:, random_index, :]
                    li_features = li_features[:, :, random_index]
                l_xyz.append(li_xyz)
                l_features.append(li_features)
        
        # filter the added relative pose
        for i in range(len(l_features)):
            li_features = l_features[i]
            if li_features is None:
                continue
            if li_features.shape[1] == 201:
                l_features[i] = torch.cat([li_features[:, 3:67,:], li_features[:, 70:134,:], li_features[:, 137:,:]], dim=1)
            if li_features.shape[1] == 393:
                l_features[i] = torch.cat([li_features[:, 3:131,:], li_features[:, 134:262,:], li_features[:, 265:,:]], dim=1)
            if li_features.shape[1] == 777:
                l_features[i] = torch.cat([li_features[:, 3:259,:], li_features[:, 262:518,:], li_features[:, 521:,:]], dim=1)
            if li_features.shape[1] == 2313:
                l_features[i] = torch.cat([li_features[:, 3:771,:], li_features[:, 774:1542,:], li_features[:, 1545:,:]], dim=1)

        _, _, global_out2_feat = self.SA_modules[5](l_xyz[3], None, l_features[3])
        
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1 - 1] = self.FP_modules[i](
                l_xyz[i - 1 - 1], l_xyz[i - 1], l_features[i - 1 - 1], l_features[i - 1]
            )
        
        cls = cls.view(-1, 16, 1).repeat(1, 1, l_features[0].size()[2])         # object class one-hot-vector
        l_features[0] = torch.cat((l_features[0], l_features[-1].repeat(1, 1, l_features[0].size()[2]), global_out2_feat.repeat(1, 1, l_features[0].size()[2]), cls), 1)
        return self.FC_layer(l_features[0]).transpose(1, 2).contiguous()
