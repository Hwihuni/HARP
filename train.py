import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # GUI window off - prevent "main thread" error
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from unet_1d import Unet_1D_tanh as Unet_1D
from loss import *
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import init_weights, init_weights_zeros
from dataset_phantom import BasicDataset
import random
import nibabel as nib
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)

def train_net(net,device,args):

    train = BasicDataset(args,'train')
    n_train = len(train)

    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=2, pin_memory=True)

    writer = SummaryWriter(comment= f'_Trn_LR_{args.lr}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batchsize}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Device:          {device.type}
    ''')

    params = [{'params':net.parameters()}]
    
    optimizer = optim.Adam(params, lr=args.lr)

    criterion = nn.MSELoss()
    
    logging.info(sum(p.numel() for p in net.parameters() if p.requires_grad))

    with tqdm(total=args.max_iter, desc=f'Step') as pbar:
        for _ in range(args.max_iter):
            for imgs, target,mask  in tqdm(train_loader):

                net.train()
                imgs = imgs.cuda(args.gpu_ind[0])
                target= target.cuda(args.gpu_ind[0])
                mask= mask.cuda(args.gpu_ind[0])
                preds_scale = nn.parallel.data_parallel(net,imgs, args.gpu_ind)
                preds = torch.zeros_like(imgs)
                preds[:,0:1,:,:] = imgs[:,0:1,:,:] * preds_scale[:,0:1,:,:]
                preds[:,1:6,:,:] = imgs[:,1:6,:,:] * preds_scale[:,1:2,:,:]
                preds[:,6:15,:,:] = imgs[:,6:15,:,:] * preds_scale[:,2:3,:,:]
                preds[:,15:28,:,:] = imgs[:,15:28,:,:] * preds_scale[:,3:4,:,:]
                preds[:,28:45,:,:] = imgs[:,28:45,:,:] * preds_scale[:,4:5,:,:]

                loss = criterion(preds[:,0:1,:,:]*mask,target[:,0:1,:,:]*mask)
                loss += criterion(preds[:,1:6,:,:]*mask,target[:,1:6,:,:]*mask)
                loss += criterion(preds[:,6:15,:,:]*mask,target[:,6:15,:,:]*mask)
                loss += criterion(preds[:,15:28,:,:]*mask,target[:,15:28,:,:]*mask)
                loss += criterion(preds[:,28:45,:,:]*mask,target[:,28:45,:,:]*mask)

                if  global_step % args.train_step == 0:
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()                          

                if global_step % args.save_step == 0:
                    torch.save(net.state_dict(),f'{args.path}/net_step_{global_step}.pth')

                if global_step >= args.max_iter:
                    break   

                global_step = global_step + 1     

                    
                pbar.update(1)
                
                         
    writer.close()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--max_iter', metavar='E', type=int, default=40000,help='Number of max_iter', dest='max_iter')
    parser.add_argument('-b', '--batchsize',  type=int, default=2,help='batch size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-6,help='Learning rate', dest='lr')
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='0',help='gpu')
    parser.add_argument('-ts', '--train_step', dest='train_step', type=int, default=100,help='train round step')
    parser.add_argument('-ss', '--save_step', dest='save_step', type=int, default=1000,help='Checkpoint saving step')
    parser.add_argument('-dp', '--data_path', dest='data_path', type=str, default='./pickle',help='Path for training data')
    parser.add_argument('-pl', '--percent_low', dest='percent_low', type=float, default=25,help='Thres value for training')
    parser.add_argument('-ph', '--percent_high', dest='percent_high', type=float, default=90,help='Thres value for training')

    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    args.path = './checkpoints'
    str_ids = args.gpu_ind.split(',')
    args.gpu_ind = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ind.append(id)

    if len(args.gpu_ind) > 0:
        torch.cuda.set_device(args.gpu_ind[0])
    args.thres_low = 1.0 # set the low thresholds based on test data
    args.thres_high = 2.0 # set the high thresholds based on test data
    logging.info(f'thres {args.thres_low} ~ {args.thres_high}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device CUDA {(args.gpu_ind)}')

    net = Unet_1D(n_channels=45, n_classes=5)
    net.cuda(args.gpu_ind[0])
    init_weights(net)
    train_net(net = net, device = device, args = args)


