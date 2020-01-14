import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from models import PointNet2_SSN_SEG as PointNet2_SSN
from data import ShapeNetPart
import utils.pytorch_utils as pt_utils
import data.data_utils as d_utils
import argparse
import random
import yaml
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123#int(time.time())
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed) 

parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Part Segmentation Training')
parser.add_argument('--config', default='cfgs/config_msn_partseg.yaml', type=str)


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
    
    train_dataset = ShapeNetPart(root=args.data_root, num_points=args.num_points, split='trainval', normalize=True)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=int(args.workers), 
        pin_memory=True
    )
    
    global test_dataset
    test_dataset = ShapeNetPart(root=args.data_root, num_points=args.num_points, split='test', normalize=True)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers), 
        pin_memory=True
    )
    global test_dataset2
    test_dataset2 = ShapeNetPart(root=args.data_root, num_points=args.num_points, split='test', normalize=True)
    test_dataloader2 = DataLoader(
        test_dataset2, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers), 
        pin_memory=True
    )
    
    model = PointNet2_SSN(num_classes=args.num_classes)
    model.cuda()
    #model = torch.nn.DataParallel(model)
    optimizer = optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)
    
    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))

    criterion = nn.CrossEntropyLoss()
    num_batch = len(train_dataset)/args.batch_size


    # training
    # train(train_dataloader, test_dataloader, test_dataloader2, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)
    validate(test_dataloader2, model, criterion, args, 1, 'so3')
    
def train(train_dataloader, test_dataloader, test_dataloader2, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch):
    global Class_mIoU, Inst_mIoU
    Class_mIoU, Inst_mIoU = 0.7, 0.7
    batch_count = 0
    aug = d_utils.SO3Rotate()
    model.train()
    for epoch in range(args.epochs):
        for i, data in enumerate(train_dataloader, 0):
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
                bnm_scheduler.step(epoch-1)

            points, norm, target, cls = data
            points, norm, target = points.cuda(), norm.cuda(), target.cuda()
            points.data, norm.data = aug(points.data, norm.data)
            
            optimizer.zero_grad()
            
            batch_one_hot_cls = np.zeros((len(cls), 16))   # 16 object classes
            for b in range(len(cls)):
                batch_one_hot_cls[b, int(cls[b])] = 1
            batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls)
            batch_one_hot_cls = batch_one_hot_cls.float().cuda()
            pred = model(points, norm, batch_one_hot_cls)
            pred = pred.view(-1, args.num_classes)
            target = target.view(-1,1)[:,0]
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            if i % args.print_freq_iter == 0:
                print('[epoch %3d: %3d/%3d] \t train loss: %0.6f \t lr: %0.5f' %(epoch+1, i, num_batch, loss.data.clone(), lr_scheduler.get_lr()[0]))
            batch_count += 1

            if args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
                # validate(test_dataloader, model, criterion, args, batch_count, 'z')
                validate(test_dataloader2, model, criterion, args, batch_count, 'so3')


def validate(test_dataloader, model, criterion, args, iter, mode): 
    global Class_mIoU, Inst_mIoU, test_dataset
    if mode == 'z':
        aug = d_utils.ZRotate()
    else:
        aug = d_utils.SO3Rotate()

    model.eval()
    
    seg_classes = test_dataset.seg_classes
    shape_ious = {cat:[] for cat in seg_classes.keys()}
    seg_label_to_cat = {}           # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    losses = []
    for j, data in enumerate(test_dataloader, 0):
        points, norm, target, cls = data
        points, norm, target = points.cuda(), norm.cuda(), target.cuda()
        points.data, norm.data = aug(points.data, norm.data)

        batch_one_hot_cls = np.zeros((len(cls), 16))   # 16 object classes
        for b in range(len(cls)):
            batch_one_hot_cls[b, int(cls[b])] = 1
        batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls)
        batch_one_hot_cls = batch_one_hot_cls.float().cuda()
        #batch_one_hot_cls = Variable(batch_one_hot_cls.float().cuda())
        with torch.no_grad():
            pred = model(points, norm, batch_one_hot_cls)
            #print(pred.shape)
            loss = criterion(pred.view(-1, args.num_classes), target.view(-1,1)[:,0])
            losses.append(loss.data.clone())
            pred = pred.data.cpu()
            target = target.data.cpu()
            pred_val = torch.zeros(len(cls), args.num_points).type(torch.LongTensor)

        # pred to the groundtruth classes (selected by seg_classes[cat])
        for b in range(len(cls)):
            cat = seg_label_to_cat[target[b, 0].item()]
            logits = pred[b, :, :]   # (num_points, num_classes)
            #print(logits[:, seg_classes[cat]].max(1)[1])
            pred_val[b, :] = logits[:, seg_classes[cat]].max(1)[1] + seg_classes[cat][0]
            
        for b in range(len(cls)):
            segp = pred_val[b, :]
            segl = target[b, :]
            

            ##############################################################
            # output = torch.cat([points[b].cpu(), segp.unsqueeze(-1).float()], dim=1).cpu().detach().numpy()
            # print(output.shape)
            # np.savetxt("segres/"+ str(b+j*len(cls))+".txt", output)
            ###########################################################

            cat = seg_label_to_cat[segl[0].item()]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                #print(segl)
                #print(seg_classes[cat])
                if torch.sum((segl == l) | (segp == l)).item() == 0:
                    # part is not present in this shape
                    part_ious[l - seg_classes[cat][0]] = 1.0
                    #print(torch.sum((segl == l) | (segp == l)))
                else:
                    #print(torch.sum((segl == l) | (segp == l)))
                    part_ious[l - seg_classes[cat][0]] = torch.sum((segl == l) & (segp == l)).item() / float(torch.sum((segl == l) | (segp == l)).item())

            shape_ious[cat].append(np.mean(part_ious))
        
    instance_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            instance_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    mean_class_ious = np.mean(list(shape_ious.values()))
    
    print(mode + ' truestart:')
    for cat in sorted(shape_ious.keys()):
        print('****** %s: %0.6f'%(cat, shape_ious[cat]))
    print('************ Test Loss: %0.6f' % (np.array(losses).mean()))
    print('************ Class_mIoU: %0.6f' % (mean_class_ious))
    print('************ Instance_mIoU: %0.6f' % (np.mean(instance_ious)))
    
    if mode != 'z':
        if mean_class_ious > Class_mIoU or np.mean(instance_ious) > Inst_mIoU:
            if mean_class_ious > Class_mIoU:
                Class_mIoU = mean_class_ious
            if np.mean(instance_ious) > Inst_mIoU:
                Inst_mIoU = np.mean(instance_ious)
            torch.save(model.state_dict(), '%s/ss_third_iter_%d_ins_%0.6f_cls_%0.6f.pth' % (args.save_path, iter, np.mean(instance_ious), mean_class_ious))
    model.train()
    
if __name__ == "__main__":
    main()
    print("so3so3_third")
