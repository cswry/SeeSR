'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
import os
import sys
sys.path.append(os.getcwd())
import cv2

import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything

import argparse
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from ram.models import ram
from ram import inference_ram as inference

parser = argparse.ArgumentParser()
parser.add_argument("--gt_path", nargs='+', default=['PATH 1', 'PATH 2'], help='the path of high-resolution images')
parser.add_argument("--save_dir", type=str, default='preset/datasets/train_datasets/training_for_seesr', help='the save path of the training dataset.')
parser.add_argument("--start_gpu", type=int, default=1, help='if you have 5 GPUs, you can set it to 1/2/3/4/5 on five gpus for parallel processing., which will save your time. ')  
parser.add_argument("--batch_size", type=int, default=10, help='smaller batch size means much time but more extensive degradation for making the training dataset.')  
parser.add_argument("--epoch", type=int, default=1, help='decide how many epochs to create for the dataset.')
args = parser.parse_args()

print(f'====== START GPU: {args.start_gpu} =========')
seed_everything(24+args.start_gpu*1000)

from torchvision.transforms import Normalize, Compose
args_training_dataset = {}

# Please set your gt path here. If you have multi dirs, you can set it as ['PATH1', 'PATH2', 'PATH3', ...]
args_training_dataset['gt_path'] = args.gt_path

#################### REALESRGAN SETTING ###########################
args_training_dataset['queue_size'] = 160
args_training_dataset['crop_size'] =  512
args_training_dataset['io_backend'] = {}
args_training_dataset['io_backend']['type'] = 'disk'

args_training_dataset['blur_kernel_size'] = 21
args_training_dataset['kernel_list'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
args_training_dataset['kernel_prob'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
args_training_dataset['sinc_prob'] = 0.1
args_training_dataset['blur_sigma'] = [0.2, 3]
args_training_dataset['betag_range'] = [0.5, 4]
args_training_dataset['betap_range'] = [1, 2]

args_training_dataset['blur_kernel_size2'] = 11
args_training_dataset['kernel_list2'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
args_training_dataset['kernel_prob2'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
args_training_dataset['sinc_prob2'] = 0.1
args_training_dataset['blur_sigma2'] = [0.2, 1.5]
args_training_dataset['betag_range2'] = [0.5, 4.0]
args_training_dataset['betap_range2'] = [1, 2]

args_training_dataset['final_sinc_prob'] = 0.8

args_training_dataset['use_hflip'] = True
args_training_dataset['use_rot'] = False

train_dataset = RealESRGANDataset(args_training_dataset)
batch_size = args.batch_size
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=11,
    drop_last=True,
)

#################### REALESRGAN SETTING ###########################
args_degradation = {}
# the first degradation process
args_degradation['resize_prob'] = [0.2, 0.7, 0.1]  # up, down, keep
args_degradation['resize_range'] = [0.15, 1.5]
args_degradation['gaussian_noise_prob'] = 0.5
args_degradation['noise_range'] = [1, 30]
args_degradation['poisson_scale_range'] = [0.05, 3.0]
args_degradation['gray_noise_prob'] = 0.4
args_degradation['jpeg_range'] = [30, 95]

# the second degradation process
args_degradation['second_blur_prob'] = 0.8
args_degradation['resize_prob2'] = [0.3, 0.4, 0.3]  # up, down, keep
args_degradation['resize_range2'] = [0.3, 1.2]
args_degradation['gaussian_noise_prob2'] = 0.5
args_degradation['noise_range2'] = [1, 25]
args_degradation['poisson_scale_range2'] = [0.05, 2.5]
args_degradation['gray_noise_prob2'] = 0.4
args_degradation['jpeg_range2'] = [30, 95]

args_degradation['gt_size']= 512
args_degradation['no_degradation_prob']= 0.01


from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian
import random
import torch.nn.functional as F

def realesrgan_degradation(batch,  args_degradation, use_usm=True, sf=4, resize_lq=True):
    jpeger = DiffJPEG(differentiable=False).cuda()
    usm_sharpener = USMSharp().cuda()  # do usm sharpening
    im_gt = batch['gt'].cuda()
    if use_usm:
        im_gt = usm_sharpener(im_gt)
    im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
    kernel1 = batch['kernel1'].cuda()
    kernel2 = batch['kernel2'].cuda()
    sinc_kernel = batch['sinc_kernel'].cuda()

    ori_h, ori_w = im_gt.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(im_gt, kernel1)
    # random resize
    updown_type = random.choices(
            ['up', 'down', 'keep'],
            args_degradation['resize_prob'],
            )[0]
    if updown_type == 'up':
        scale = random.uniform(1, args_degradation['resize_range'][1])
    elif updown_type == 'down':
        scale = random.uniform(args_degradation['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = args_degradation['gray_noise_prob']
    if random.random() < args_degradation['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=args_degradation['noise_range'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=args_degradation['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range'])
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if random.random() < args_degradation['second_blur_prob']:
        out = filter2D(out, kernel2)
    # random resize
    updown_type = random.choices(
            ['up', 'down', 'keep'],
            args_degradation['resize_prob2'],
            )[0]
    if updown_type == 'up':
        scale = random.uniform(1, args_degradation['resize_range2'][1])
    elif updown_type == 'down':
        scale = random.uniform(args_degradation['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
            out,
            size=(int(ori_h / sf * scale),
                    int(ori_w / sf * scale)),
            mode=mode,
            )
    # add noise
    gray_noise_prob = args_degradation['gray_noise_prob2']
    if random.random() < args_degradation['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=args_degradation['noise_range2'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=args_degradation['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False,
            )

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if random.random() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(ori_h // sf,
                        ori_w // sf),
                mode=mode,
                )
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(ori_h // sf,
                        ori_w // sf),
                mode=mode,
                )
        out = filter2D(out, sinc_kernel)

    # clamp and round
    im_lq = torch.clamp(out, 0, 1.0)

    # random crop
    gt_size = args_degradation['gt_size']
    im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, sf)
    lq, gt = im_lq, im_gt


    gt = torch.clamp(gt, 0, 1)
    lq = torch.clamp(lq, 0, 1)

    return lq, gt


root_path = args.save_dir
gt_path = os.path.join(root_path, 'gt')
lr_path = os.path.join(root_path, 'lr')
sr_bicubic_path = os.path.join(root_path, 'sr_bicubic')
os.makedirs(gt_path, exist_ok=True)
os.makedirs(lr_path, exist_ok=True)
os.makedirs(sr_bicubic_path, exist_ok=True)


epochs = args.epoch
step = len(train_dataset) * epochs * args.start_gpu
with torch.no_grad():
    for epoch in range(epochs):
        for num_batch, batch in enumerate(train_dataloader):
            lr_batch, gt_batch = realesrgan_degradation(batch, args_degradation=args_degradation)
            sr_bicubic_batch = F.interpolate(lr_batch, size=(gt_batch.size(-2), gt_batch.size(-1)), mode='bicubic',)

            for i in range(batch_size):
                step += 1
                print('process {} images...'.format(step))
                lr = lr_batch[i, ...]
                gt = gt_batch[i, ...]
                sr_bicubic = sr_bicubic_batch[i, ...]

                lr_save_path =  os.path.join(lr_path,'{}.png'.format(str(step).zfill(7)))
                gt_save_path =  os.path.join(gt_path, '{}.png'.format(str(step).zfill(7)))
                sr_bicubic_save_path =  os.path.join(sr_bicubic_path, '{}.png'.format(str(step).zfill(7)))

                cv2.imwrite(lr_save_path, 255*lr.detach().cpu().squeeze().permute(1,2,0).numpy()[..., ::-1])
                cv2.imwrite(gt_save_path, 255*gt.detach().cpu().squeeze().permute(1,2,0).numpy()[..., ::-1])
                cv2.imwrite(sr_bicubic_save_path, 255*sr_bicubic.detach().cpu().squeeze().permute(1,2,0).numpy()[..., ::-1])
               

            del lr_batch, gt_batch, sr_bicubic_batch
            torch.cuda.empty_cache()
    
