from os.path import splitext
from os import listdir
import numpy as np 
import os
import scipy.io
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import fastmri
class BasicDataset(Dataset):
    def __init__(self, args,mode):
        self.args = args
        self.mode = mode
        path_pckl_diff_src = f'{args.data_path}/pkl_bwh_invivo_ge_siemens_GE_sphhar_norm1000.pklv4'
        path_pckl_diff_trg = f'{args.data_path}/pkl_bwh_invivo_ge_siemens_Siemens_sphhar_norm1000.pklv4'
        path_mask = f'{args.data_path}/pkl_bwh_invivo_ge_siemens_GE_mask.pklv4'

        self.target_diff = []
        self.source_diff = []
        self.mask = []

        with open(path_pckl_diff_trg, "rb") as f:
            self.target_diff += pickle.load(f)

        with open(path_pckl_diff_src, "rb") as f:
            self.source_diff += pickle.load(f)

        with open(path_mask, "rb") as f:
            self.mask += pickle.load(f)


        self.target_diff = [np.transpose((img), [2, 0, 1]) for img in self.target_diff]
        self.source_diff = [np.transpose((img), [2, 0, 1]) for img in self.source_diff]
        self.mask = [np.transpose((img), [2, 0, 1]) for img in self.mask]
            
        self.target_diff = [img*(mask) for img,mask in zip(self.target_diff,self.mask)]
        self.source_diff = [img*(mask) for img,mask in zip(self.source_diff,self.mask)]
            
        logging.info(f'Creating dataset with {len(self.target_diff)} examples')

        # self.scale = [np.expand_dims(img,0) for img in self.scale]

        
    def load_pkls( self,path):
        assert os.path.isfile(path), path
        images = []
        with open(path, "rb") as f:
            images += pickle.load(f)
        assert len(images) > 0, path
        return images

    def __len__(self):
        return len(self.target_diff)
    
    def __getitem__(self, i):
        source_i = self.source_diff[i]
        target_i = self.target_diff[i]
        assert source_i.size == target_i.size, \
            f'Image and mask {i} should be the same size, but are {source_i.size} and {target_i.size}'
           
    
        return source_i.astype('float32'), target_i.astype('float32'),self.mask[i]
def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x,axes = (-2,-1)),norm='ortho'),axes = (-2,-1))

def fft2c(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(x,axes = (-2,-1)),norm='ortho'),axes = (-2,-1))
