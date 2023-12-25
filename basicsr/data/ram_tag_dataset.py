import cv2
import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np
import math

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from PIL import Image


@DATASET_REGISTRY.register()
class RAMTagDataset(Dataset):
    def __init__(self, opt, image_size=384): 
        
        self.opt = opt
        self.root = opt['root']
        exts = opt['ext']

        gt_lists = []
        lr_lists = []
        for idx_dir, root_dir in enumerate(self.root):
            gt_path = os.path.join(root_dir, 'gt')
            lr_path = os.path.join(root_dir, 'sr_bicubic')
            print(f'gt_path: {gt_path}')
            for ext in exts:
                gt_list = glob.glob(os.path.join(gt_path, ext))
                lr_list = glob.glob(os.path.join(lr_path, ext))
                gt_lists += gt_list
                lr_lists += lr_list

        self.lr_lists = lr_lists
        self.gt_lists = gt_lists
        
        assert len(self.gt_lists) == len(self.lr_lists)

        print(f'=========================Dataset Length {len(self.gt_lists)}=========================')

        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
        ])

        self.ram_preproc = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        gt_image = Image.open(self.gt_lists[index]).convert('RGB') 
        lr_image = Image.open(self.lr_lists[index]).convert('RGB') 

        lr_image, gt_image = self.img_preproc(lr_image), self.img_preproc(gt_image)
        lr_image_ram, gt_image_ram = self.ram_preproc(lr_image), self.ram_preproc(gt_image)
        return_d = {'gt': gt_image, 'lq': lr_image, 'gt_ram': gt_image_ram, 'lq_ram': lr_image_ram, 'lq_path':self.lr_lists[index]}
        return return_d
        

    def __len__(self):
        return len(self.gt_lists)


        
