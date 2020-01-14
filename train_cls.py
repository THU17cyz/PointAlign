import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from models import PointNet2_SSN_CLS as PointNet2_SSN
from models import RSCNN_SSN_CLS as RSCNN_SSN
from models import RSCNN_MSN_CLS as RSCNN_MSN
from data import ModelNet40Cls
import utils.pytorch_utils as pt_utils
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
import argparse
import random
import yaml
import time

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed) 


os.environ['CUDA_VISIBLE_DEVICES'] = '7'

parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Classification Training')
parser.add_argument('--config', default='cfgs/pointnet2_config_ssn_cls.yaml', type=str)


def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")
    
    try:
        os.makedirs(args.save_path)
    except OSError:
        pass
    
    train_dataset = ModelNet40Cls(num_points=args.num_points, root=args.data_root, transforms=None)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=int(args.workers), 
        pin_memory=True
    )

    test_dataset_z = ModelNet40Cls(num_points=args.num_points, root=args.data_root, transforms=None, train=False)
    test_dataloader_z = DataLoader(
        test_dataset_z,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers), 
        pin_memory=True
    )

    test_dataset_so3 = ModelNet40Cls(num_points=args.num_points, root=args.data_root, transforms=None, train=False)
    test_dataloader_so3 = DataLoader(
        test_dataset_so3,
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers), 
        pin_memory=True
    )
    if args.model == "pointnet2_ssn":
        model = PointNet2_SSN(num_classes=args.num_classes)
        model.cuda()
        # model = torch.nn.DataParallel(model)
    elif args.model == "rscnn_ssn":
        model = RSCNN_SSN(num_classes=args.num_classes)
        model.cuda()
        model = torch.nn.DataParallel(model)
    elif args.model == "rscnn_msn":
        model = RSCNN_MSN(num_classes=args.num_classes)
        model.cuda()
        model = torch.nn.DataParallel(model)
    else:
        print("Doesn't support this model")
        return
    model.cuda()
    #model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)
    
    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % args.checkpoint)

    criterion = nn.CrossEntropyLoss()
    num_batch = len(train_dataset)/args.batch_size
    
    # training
    # train(train_dataloader, test_dataloader_z, test_dataloader_so3, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)
    validate(test_dataloader_so3, model, criterion, args, 0, 'so3')

def train(train_dataloader,
          test_dataloader_z,
          test_dataloader_so3,
          model,
          criterion,
          optimizer,
          lr_scheduler,
          bnm_scheduler,
          args,
          num_batch):
    aug = d_utils.ZRotate()
    global g_acc 
    g_acc = 0.88    # only save the model whose acc > 0.91
    batch_count = 0
    model.train()
    for epoch in range(args.epochs):
        for i, data in enumerate(train_dataloader, 0):
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
                bnm_scheduler.step(epoch-1)
            points, normals, target = data
            points, normals, target = points.cuda(), normals.cuda(), target.cuda()
            if args.model == "pointnet2":
                fps_idx = pointnet2_utils.furthest_point_sample(points, 1024)  # (B, npoint)
                points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                normals = pointnet2_utils.gather_operation(normals.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
            else:
                fps_idx = pointnet2_utils.furthest_point_sample(points, 1200)  # (B, npoint)
                fps_idx = fps_idx[:, np.random.choice(1200, args.num_points, False)]
                points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                normals = pointnet2_utils.gather_operation(normals.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
                # augmentation
                points.data = d_utils.PointcloudScaleAndTranslate()(points.data)  # not scale and translate

            points.data, normals.data = aug(points.data, normals.data)

            optimizer.zero_grad()
            pred = model(points, normals)
            target = target.view(-1)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            if i % args.print_freq_iter == 0:
                print('[epoch %3d: %3d/%3d] \t train loss: %0.6f \t lr: %0.5f' %(epoch+1, i, num_batch, loss.data.clone(), lr_scheduler.get_lr()[0]))
            batch_count += 1
            
            # validation in between an epoch
            if args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
                # validate(test_dataloader_z, model, criterion, args, batch_count, 'z')
                validate(test_dataloader_so3, model, criterion, args, batch_count, 'so3')


def validate(test_dataloader, model, criterion, args, iter, mode): 
    global g_acc
    if mode == 'z':
        aug = d_utils.ZRotate()
    else:
        aug = d_utils.SO3Rotate()
    model.eval()
    losses, preds, labels = [], [], []
    for j, data in enumerate(test_dataloader, 0):
        points, normals, target = data
        points, normals, target = points.cuda(), normals.cuda(), target.cuda()
        
        # fastest point sampling
        fps_idx = pointnet2_utils.furthest_point_sample(points, args.num_points)  # (B, npoint)
        # fps_idx = fps_idx[:, np.random.choice(1200, args.num_points, False)]
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
        normals = pointnet2_utils.gather_operation(normals.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

        points.data, normals.data = aug(points.data, normals.data)

        with torch.no_grad():
            pred = model(points, normals)
            target = target.view(-1)
            loss = criterion(pred, target)
            losses.append(loss.data.clone())
            _, pred_choice = torch.max(pred.data, -1)
        
            preds.append(pred_choice)
            labels.append(target.data)
        
    preds = torch.cat(preds, 0)
    labels = torch.cat(labels, 0)
    acc = (preds == labels).sum().item() / labels.numel()
    print(acc)
    if mode == 'z':
        print('z val loss: %0.6f \t acc: %0.6f\n' %(np.array(losses).mean(), acc))
    else:
        print('so3 val loss: %0.6f \t acc: %0.6f\n' %(np.array(losses).mean(), acc))
    if acc > g_acc and mode != 'z':
        g_acc = acc
        torch.save(model.state_dict(), '%s/zso3ours_iter_%d_acc_%0.6f.pth' % (args.save_path, iter, acc))
    model.train()
    #assert 1 > 2


if __name__ == "__main__":
    print("mynorm48, scaleaug and random fps, ssn, 16, put norm/dir in h, and don't use xyz in second layer, max pooling!")
    print("zso3_ours")
    main()
    print("mynorm48, scaleaug and random fps, ssn, 16, put norm/dir in h, and don't use xyz in second layer, max pooling!")
