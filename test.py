import argparse
import logging
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from unet_1d import Unet_1D_tanh as Unet_1D
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import QballModel

import nibabel as nib
from torch.utils.data import DataLoader
from dataset_invivo import BasicDataset
import random
import pickle
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.autograd.set_detect_anomaly(True)

def create_all_dirs(path):
    if "." in path.split("/")[-1]:
        dirs = os.path.dirname(path)
    else:
        dirs = path
    os.makedirs(dirs, exist_ok=True)

def to_pklv4(obj, path, vebose=False):
    create_all_dirs(path)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    if vebose:
        print("Wrote {}".format(path))

def test_net(val_loader,net,device,args):


    
    logging.info(sum(p.numel() for p in net.parameters() if p.requires_grad))

    net.eval()

    recons = []
    ind = 0
    inputs = []
    slice_nums = np.array([0,90,180,270,360,450])
    for imgs_val, target_val,mask  in tqdm(val_loader):

        imgs_val = imgs_val.cuda(args.gpu_ind[0])
        target_val= target_val.cuda(args.gpu_ind[0])
        mask= mask.cuda(args.gpu_ind[0])
        with torch.no_grad():                  
            preds_scale = nn.parallel.data_parallel(net,imgs_val, args.gpu_ind)
            preds = torch.zeros_like(imgs_val)
            preds[:,0:1,:,:] = imgs_val[:,0:1,:,:] * preds_scale[:,0:1,:,:]* mask
            preds[:,1:6,:,:] = imgs_val[:,1:6,:,:] * preds_scale[:,1:2,:,:]* mask
            preds[:,6:15,:,:] = imgs_val[:,6:15,:,:] * preds_scale[:,2:3,:,:]* mask
            preds[:,15:28,:,:] = imgs_val[:,15:28,:,:] * preds_scale[:,3:4,:,:]* mask
            preds[:,28:45,:,:] = imgs_val[:,28:45,:,:] * preds_scale[:,4:5,:,:]* mask
            im = preds.cpu().detach().numpy()

        for i in range(im.shape[0]):
            
            sub_num = np.where((ind-2*slice_nums)>=0)[0][-1]
            sess_num = (ind-2*slice_nums[sub_num]) // (slice_nums[sub_num+1]-slice_nums[sub_num])
            slice_ind = (ind-2*slice_nums[sub_num]) % (slice_nums[sub_num+1]-slice_nums[sub_num])
            if slice_ind == 0:
                bval = f'./images/GE_{sub_num+1}/GE_{sub_num+1}_{sess_num+1}_dwi_AP.bval'
                bvec = f'./images/GE_{sub_num+1}/combined_ed_GE_{sub_num+1}_{sess_num+1}.eddy_rotated_bvecs'
                b0 = nib.load(f'./images/GE_{sub_num+1}/combined_ed_GE_{sub_num+1}_{sess_num+1}.nii.gz').get_fdata()
                print(bvec)
                bvals, bvecs = read_bvals_bvecs(bval, bvec)
                gtab = gradient_table(bvals, bvecs, b0_threshold=50)
                qb_model = QballModel(gtab, sh_order=8)
                img = np.zeros((146,146,slice_nums[sub_num+1]-slice_nums[sub_num],62))

            inp = np.transpose(imgs_val[i,...].cpu().detach().numpy(),[1,2,0])

            recon = np.abs(np.concatenate((b0[:,:,slice_ind,:2],(np.dot(np.transpose(im[i,...],[1,2,0]),qb_model.B.T) **  (bvals.max()/1000.0)) * np.mean(b0[:,:,slice_ind:slice_ind+1,:2],-1)),-1)).astype('float32')
            img[:,:,slice_ind,:] = recon
            if slice_ind == (slice_nums[sub_num+1]-slice_nums[sub_num])-1:
                recons.append(img)
            ind = ind + 1

    return recons,inputs
        

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
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=500,help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batchsize',  type=int, default=2,help='batch size')
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='0',help='gpu')
    parser.add_argument('--load',  type=int, default=0,help='Checkpoint load')
    parser.add_argument('-dp', '--data_path', dest='data_path', type=str, default='./pickle',help='Path for training data')
    parser.add_argument('-op', '--option', dest='option', type=str, default="Trn_scale01_norm1000_weighted_mask_high_95.0_low_30.0_LR_1e-06")


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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device CUDA {(args.gpu_ind)}')

    val = BasicDataset(args,'test')
    
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    net = Unet_1D(n_channels=45, n_classes=5)

    net.cuda(args.gpu_ind[0])
    step = 40000
    load = f'./checkpoints/net_step_{step}.pth'

    os.makedirs(f'./inference/GE_har',exist_ok=True)

    print(load)
    net.load_state_dict(torch.load(load,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
    recon,_  = test_net(val_loader=val_loader,net = net, device = device, args = args)

    names_val = [f'{i}_{j}' for i in range(1,6) for j in range(1,3)]
    for ind,name in enumerate(names_val):
        tmp = nib.load(f'./images/GE_{name[0]}/combined_ed_GE_{name}.nii.gz')
        os.makedirs(f'./inference/GE_har/{name}',exist_ok=True)

        img_path_dwi = f'./inference/GE_har/{name}/dwi_GE_har_{name}.nii.gz'
        img_dwi = recon[ind]
        nib.Nifti1Image(img_dwi, tmp.affine,  tmp.header).to_filename(img_path_dwi)



