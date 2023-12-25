'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Mapping, Any

import random
import os
import cv2
import glob
import json
import math
from tqdm import tqdm


import numpy as np
from PIL import Image

import sys
sys.path.append(os.getcwd())

from ram.models.ram import ram
from ram import inference_ram as inference
from ram import get_transform
from ram.utils import build_openset_label_embedding

from basicsr.data.ram_tag_dataset import RAMTagDataset

ram_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default='preset/datasets/train_datasets/training_for_seesr', help='the dataset you want to tag.') # 
parser.add_argument("--start_gpu", type=int, default=0, help='if you have 5 GPUs, you can set it to 0/1/2/3/4 when using different GPU for parallel processing. It will save your time.') 
parser.add_argument("--all_gpu", type=int, default=1, help='if you set --start_gpu max to 5, please set it to 5') 
args = parser.parse_args()

gt_path = os.path.join(args.root_path, 'gt')
tag_path = os.path.join(args.root_path, 'tag')
os.makedirs(tag_path, exist_ok=True)

lq_lists = glob.glob(os.path.join(gt_path, '*.png'))
print(f'There are {len(lq_lists)} imgs' )

model = ram(pretrained='preset/models/ram_swin_large_14m.pth',   
                            image_size=384,
                            vit='swin_l')
model = model.eval()
model = model.to('cuda')

start_num = args.start_gpu * len(lq_lists)//args.all_gpu
end_num = (args.start_gpu+1) * len(lq_lists)//args.all_gpu

print(f'===== process [{start_num}   {end_num}] =====')

with torch.no_grad():
    for lq_idx, lq_path in enumerate(lq_lists[start_num:end_num]):
        print(f' ====== process {lq_idx} imgs... =====')
        basename = os.path.basename(lq_path).split('.')[0]
        lq = ram_transforms(Image.open(lq_path)).unsqueeze(0).to('cuda')
        gt_captions = inference(lq, model)
        gt_prompt = f"{gt_captions[0]},"
        tag_save_path = tag_path + f'{basename}.txt'
        f = open(f"{tag_save_path}", "w")
        f.write(gt_prompt)
        f.close()
        print(f'The GT tag of {basename}.txt: {gt_prompt}')












        
