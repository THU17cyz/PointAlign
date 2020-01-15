import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.autograd.function import InplaceFunction
from itertools import repeat
import numpy as np
import shutil
import os
from typing import List, Tuple
from scipy.stats import t as student_t
import statistics as stats
import math

from utils.pytorch_utils.point_align import PointAlign


def unitization(vec):
    """
    turns the input vector to a unitized one
    """
    vec_len = torch.norm(vec, dim=2).unsqueeze(-1)
    B, N = vec_len.size()[0], vec_len.size()[1]
    tiny = torch.Tensor([1e-8]).unsqueeze(0).unsqueeze(0).expand(B, N, 1).cuda()
    vec_len = vec_len + tiny
    vec = vec / vec_len
    return vec


class ConvLayer(nn.Module):
    """
    Input shape: (B, C_in, npoint, nsample)
    Output shape: (B, C_out, npoint)
    """
    def __init__(self, mlp=None, first_layer=False, last_layer=False):
        super(ConvLayer, self).__init__()
        self.first_layer = first_layer
        self.mlp = mlp
        self.last_layer = last_layer
        
    def forward(self, in_fts):
        """
        in_fts shape: (B, 3 + 3 + 3 + C_in, npoint, centroid + nsample)
        3 + 3 + 3 + C_in --> absolute coords, normal vectors, relative coords, input features
        """
        input, normal = in_fts[0], in_fts[1]
        if self.first_layer:
            x = input[:, 6:, :, :]
        else:
            x = torch.cat([input[:, 6:9, :, :], input[:, 12:, :, :]], dim=1)
        nsample = x.size()[3]

        other_normal = input[:, 3:6, :, :]
        delta_x = input[:, 6:9, :, :]    # (B, 3, npoint, nsample+1), normalized coordinates

        # azimuthal vector is the average of all the projection of relative coords on xy plane
        azi_vec = torch.mean(delta_x[:, :, :, 1:], 3).transpose(1, 2).contiguous()
        point_align = PointAlign(normal, azi_vec, nsample)
        other_normal = point_align.align(other_normal.transpose(1,3).contiguous().unsqueeze(-1))

        # if not first layer, calculate the relative pose
        if not self.first_layer:
            dir_difs = []
            for i in range(1):
                other_dir = input[:, 9:12, :, :]
                azi_vec = unitization(azi_vec)
                dir_dif = azi_vec.unsqueeze(1) - other_dir.transpose(1,3).contiguous()
                dir_dif = point_align.align(dir_dif.unsqueeze(-1))
                dir_difs.append(dir_dif)
            dir_difs = torch.mean(torch.stack(dir_difs, dim=0), dim=0)

        # x is cord info, so has to be rotate invariant too
        tmp = x[:, 0:3, :, :]
        tmp = point_align.align(tmp.transpose(1,3).contiguous().unsqueeze(-1))
        x[:, 0:3, :, :] = tmp

        # concatenate features with relative normal and relative pose information
        if self.first_layer:
            x = torch.cat([x, other_normal], dim=1)
        else:
            x = torch.cat([x, other_normal, dir_difs], dim=1)

        x = self.mlp(x)
        x = F.max_pool2d(x, kernel_size = (1, nsample)).squeeze(3)

        # if not last layer, concatenate the relative pose features
        if not self.last_layer:
            azi_vec = unitization(azi_vec)
            x = torch.cat([azi_vec.transpose(1, 2).contiguous(), x], dim=1)
        return x


class PointNetConv(nn.Sequential):
    def __init__(self, mlp=None, first_layer=False, last_layer=False
    ):
        super().__init__()

        self.add_module('Conv{}'.format(0), ConvLayer(
                mlp=mlp,
                first_layer=first_layer,
                last_layer=last_layer
        ))


class GloAvgConv(nn.Module):
    """
    Input shape: (B, C_in, 1, nsample)
    Output shape: (B, C_out, npoint)
    """
    def __init__(self, mlp):
        super(GloAvgConv, self).__init__()
        self.mlp = mlp
        
    def forward(self, x):
        x = x[0]
        nsample = x.size()[3]
        x = self.mlp(x)
        x = F.max_pool2d(x, kernel_size=(1, nsample)).squeeze(3)
        return x


class SharedMLP(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = ""
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact
                )
            )


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant(self[0].weight, 1.0)
        nn.init.constant(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class BatchNorm3d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name=""
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)


class Conv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name
        )


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name
        )


class Conv3d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int, int] = (1, 1, 1),
            stride: Tuple[int, int, int] = (1, 1, 1),
            padding: Tuple[int, int, int] = (0, 0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv3d,
            batch_norm=BatchNorm3d,
            bias=bias,
            preact=preact,
            name=name
        )


class FC(nn.Sequential):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=None,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)


class _DropoutNoScaling(InplaceFunction):

    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        if inplace:
            return None
        n = g.appendNode(
            g.create("Dropout", [input]).f_("ratio",
                                            p).i_("is_test", not train)
        )
        real = g.appendNode(g.createSelect(n, 0))
        g.appendNode(g.createSelect(n, 1))
        return real

    @classmethod
    def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, "
                "but got {}".format(p)
            )
        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = cls._make_noise(input)
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise.expand_as(input)
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(Variable(ctx.noise)), None, None, None
        else:
            return grad_output, None, None, None


dropout_no_scaling = _DropoutNoScaling.apply


class _FeatureDropoutNoScaling(_DropoutNoScaling):

    @staticmethod
    def symbolic(input, p=0.5, train=False, inplace=False):
        return None

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(
            input.size(0), input.size(1), *repeat(1,
                                                  input.dim() - 2)
        )


feature_dropout_no_scaling = _FeatureDropoutNoScaling.apply


def group_model_params(model: nn.Module):
    decay_group = []
    no_decay_group = []

    for name, param in model.named_parameters():
        if name.find("bn") != -1 or name.find("bias") != -1:
            no_decay_group.append(param)
        else:
            decay_group.append(param)

    assert len(list(model.parameters())
              ) == len(decay_group) + len(no_decay_group)

    return [
        dict(params=decay_group),
        dict(params=no_decay_group, weight_decay=0.0)
    ]


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        'epoch': epoch,
        'best_prec': best_prec,
        'model_state': model_state,
        'optimizer_state': optim_state
    }


def save_checkpoint(
        state, is_best, filename='checkpoint', bestname='model_best'
):
    filename = '{}.pth.tar'.format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}.pth.tar'.format(bestname))


def load_checkpoint(model=None, optimizer=None, filename='checkpoint'):
    filename = "{}.pth.tar".format(filename)
    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        print("==> Done")
    else:
        print("==> Checkpoint '{}' not found".format(filename))

    return epoch, best_prec


def variable_size_collate(pad_val=0, use_shared_memory=True):
    import collections
    _numpy_type_map = {
        'float64': torch.DoubleTensor,
        'float32': torch.FloatTensor,
        'float16': torch.HalfTensor,
        'int64': torch.LongTensor,
        'int32': torch.IntTensor,
        'int16': torch.ShortTensor,
        'int8': torch.CharTensor,
        'uint8': torch.ByteTensor,
    }

    def wrapped(batch):
        "Puts each data field into a tensor with outer dimension batch size"

        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
        elem_type = type(batch[0])
        if torch.is_tensor(batch[0]):
            max_len = 0
            for b in batch:
                max_len = max(max_len, b.size(0))

            numel = sum([int(b.numel() / b.size(0) * max_len) for b in batch])
            if use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            else:
                out = batch[0].new(numel)

            out = out.view(
                len(batch), max_len,
                *[batch[0].size(i) for i in range(1, batch[0].dim())]
            )
            out.fill_(pad_val)
            for i in range(len(batch)):
                out[i, 0:batch[i].size(0)] = batch[i]

            return out
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg.format(elem.dtype))

                return wrapped([torch.from_numpy(b) for b in batch])
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return _numpy_type_map[elem.dtype.name](
                    list(map(py_type, batch))
                )
        elif isinstance(batch[0], int):
            return torch.LongTensor(batch)
        elif isinstance(batch[0], float):
            return torch.DoubleTensor(batch)
        elif isinstance(batch[0], collections.Mapping):
            return {key: wrapped([d[key] for d in batch]) for key in batch[0]}
        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [wrapped(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

    return wrapped


class TrainValSplitter():
    r"""
        Creates a training and validation split to be used as the sampler in a pytorch DataLoader
    Parameters
    ---------
        numel : int
            Number of elements in the entire training dataset
        percent_train : float
            Percentage of data in the training split
        shuffled : bool
            Whether or not shuffle which data goes to which split
    """

    def __init__(
            self, *, numel: int, percent_train: float, shuffled: bool = False
    ):
        indicies = np.array([i for i in range(numel)])
        if shuffled:
            np.random.shuffle(indicies)

        self.train = torch.utils.data.sampler.SubsetRandomSampler(
            indicies[0:int(percent_train * numel)]
        )
        self.val = torch.utils.data.sampler.SubsetRandomSampler(
            indicies[int(percent_train * numel):-1]
        )


class CrossValSplitter():
    r"""
        Class that creates cross validation splits.  The train and val splits can be used in pytorch DataLoaders.  The splits can be updated
        by calling next(self) or using a loop:
            for _ in self:
                ....
    Parameters
    ---------
        numel : int
            Number of elements in the training set
        k_folds : int
            Number of folds
        shuffled : bool
            Whether or not to shuffle which data goes in which fold
    """

    def __init__(self, *, numel: int, k_folds: int, shuffled: bool = False):
        inidicies = np.array([i for i in range(numel)])
        if shuffled:
            np.random.shuffle(inidicies)

        self.folds = np.array(np.array_split(inidicies, k_folds), dtype=object)
        self.current_v_ind = -1

        self.val = torch.utils.data.sampler.SubsetRandomSampler(self.folds[0])
        self.train = torch.utils.data.sampler.SubsetRandomSampler(
            np.concatenate(self.folds[1:], axis=0)
        )

        self.metrics = {}

    def __iter__(self):
        self.current_v_ind = -1
        return self

    def __len__(self):
        return len(self.folds)

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self)
        self.val.inidicies = self.folds[idx]
        self.train.inidicies = np.concatenate(
            self.folds[np.arange(len(self)) != idx], axis=0
        )

    def __next__(self):
        self.current_v_ind += 1
        if self.current_v_ind >= len(self):
            raise StopIteration

        self[self.current_v_ind]

    def update_metrics(self, to_post: dict):
        for k, v in to_post.items():
            if k in self.metrics:
                self.metrics[k].append(v)
            else:
                self.metrics[k] = [v]

    def print_metrics(self):
        for name, samples in self.metrics.items():
            xbar = stats.mean(samples)
            sx = stats.stdev(samples, xbar)
            tstar = student_t.ppf(1.0 - 0.025, len(samples) - 1)
            margin_of_error = tstar * sx / sqrt(len(samples))
            print("{}: {} +/- {}".format(name, xbar, margin_of_error))


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)