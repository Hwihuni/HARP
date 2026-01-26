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
    def __init__(self,args,mode):
        self.args = args
        self.mode = mode
        path_trg_phantom_real = f'{args.data_path}/pkl_bwh_phantom_ge_siemens_Siemens_sphhar_norm1000.pklv4'
        path_mask = f'{args.data_path}/pkl_bwh_phantom_ge_siemens_Siemens_mask.pklv4'
        path_src_phantom_real = f'{args.data_path}/pkl_bwh_phantom_ge_siemens_GE_registered_sphhar_norm1000.pklv4'

        self.source_phantom = []
        self.target_phantom = []
        self.mask = []
        self.thres_low = args.thres_low
        self.thres_high = args.thres_high
        with open(path_trg_phantom_real, "rb") as f:
            self.target_phantom += pickle.load(f)

        with open(path_src_phantom_real, "rb") as f:
            self.source_phantom += pickle.load(f)

        with open(path_mask, "rb") as f:
            self.mask += pickle.load(f)
        
        self.target_phantom = [np.transpose((img), [2, 0, 1]) for img in self.target_phantom]
        self.source_phantom = [np.transpose((img), [2, 0, 1]) for img in self.source_phantom]
        self.mask = [np.transpose((img), [2, 0, 1]) for img in self.mask]
        
            
        self.target_phantom = [img*(mask) for img,mask in zip(self.target_phantom,self.mask)]
        self.source_phantom = [img*(mask) for img,mask in zip(self.source_phantom,self.mask)]

        logging.info(f'Creating dataset with {len(self.target_phantom)} examples')


    # @classmethod    
    def __len__(self):
        return len(self.target_phantom)
    
    def __getitem__(self, i):
        source_i = self.source_phantom[i]
        target_i = self.target_phantom[i]
        mask =self.mask[i] * (source_i[0,...] > self.thres_low) * (source_i[0,...] < self.thres_high)* (target_i[0,...] > self.thres_low) * (target_i[0,...] < self.thres_high)
        assert source_i.size == target_i.size, \
            f'Image and mask {i} should be the same size, but are {source_i.size} and {target_i.size}'
           
    
        return source_i.astype('float32'), target_i.astype('float32'), mask


def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x,axes = (-2,-1)),norm='ortho'),axes = (-2,-1))

def fft2c(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.ifftshift(x,axes = (-2,-1)),norm='ortho'),axes = (-2,-1))


