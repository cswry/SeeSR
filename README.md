<div align=center class="logo">
      <img src="figs/logo1.png" style="width:640px">
   </a>
</div>

      
## SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 


Codes and pretained models will be released soon. 

<a href='https://arxiv.org/abs/2311.16518'><img src='https://img.shields.io/badge/arXiv-2311.16518-b31b1b.svg'></a> &nbsp;&nbsp;

[Rongyuan Wu](https://scholar.google.com.hk/citations?hl=zh-CN&user=A-U8zE8AAAAJ)<sup>1,2</sup> | [Tao Yang](https://cg.cs.tsinghua.edu.cn/people/~tyang/)<sup>3</sup> | [Lingchen Sun](https://scholar.google.com/citations?hl=zh-CN&tzom=-480&user=ZCDjTn8AAAAJ)<sup>1,2</sup> | [Zhengqiang Zhang](https://scholar.google.com.hk/citations?hl=zh-CN&user=UX26wSMAAAAJ)<sup>1,2</sup> | [Shuai Li](https://scholar.google.com.hk/citations?hl=zh-CN&user=Bd73ldQAAAAJ)<sup>1,2</sup> | [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/)<sup>1,2</sup>

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute, <sup>3</sup>ByteDance Inc. 


:star: If SeeSR is helpful to your images or projects, please help star this repo. Thanks! :hugs:
### 📢 News
- **2023.12.25**: Release SeeSR-SD2.1Base. *Merry Christmas!* 🎅🎄🎅🎄
- **2023.11.28**: Create this repo.

### 📌 TODO
- [ ] SeeSR-SDXL
- [ ] SeeSR-SD2.1Base-face,text
- [ ] SeeSR Acceleration

## 🔎 Overview framework:
![seesr](figs/framework.png)

## 📷 Real-World Results
![seesr](figs/data_real_suppl.jpg)

## ⚙️ Dependencies and Installation
```
## git clone this repository
git clone https://github.com/cswry/SeeSR.git
cd SeeSR

# create an environment with python >= 3.8
conda create -n seesr python=3.8
conda activate seesr
pip install -r requirements.txt
```

## 🚀 Quick Inference
##### Download the pretrained models
Download the pretrained SD-2.1base models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) and the SeeSR models from [GoogleDrive](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) or [BaiduDrive](https://huggingface.co/stabilityai/stable-diffusion-2-1-base). You can put the models into `preset/models`.

#### Prepare testing data
You can put the testing images in the `preset/datasets/test_datasets`

#### Running testing command
```
python test_seesr.py \
--pretrained_model_path preset/models/stable-diffusion-2-base \
--prompts None \
--seesr_model_path preset/models/SeeSR_model \
--ram_ft_path preset/models/DAPE.pth \
--image_path preset/datasets/test_datasets \
--output_dir preset/datasets/output \
--start_point lr \ 
--num_inference_steps 50 \
--guidance_scale 6.5 \
--process_size 512 
```

The default settings are optimized for the best result. However, the behavior of the code can be customized:
- Trade-offs between the **fidelity** and **perception**  
  - `--num_inference_steps`: Using more sampling steps in `Real-world SR` tasks is not a purely beneficial choice. While it improves the perception quality, it can also reduce fidelity quality as it generates more. Considering the trade-offs between fidelity and perception, as well as the inference time cost, we set the default value to `50`. However, you can make appropriate adjustments based on your specific needs.
  - `--guidance_scale`: A higher value means unleashing more generation capacity of SD, which improves perception quality but decreases fidelity quality. We set the default value to `6.5`, you can make appropriate adjustments based on your specific needs.
  - `--process_size`: The inference script resizes input images to the `process_size`, and then resizes the prediction back to the original resolution after process. We found that increasing the processing size (e.g. 768) improves fidelity but decreases perception. We set the default value to `512`, consistent with the training size of the pre-trained SD model. You can make appropriate adjustments based on your specific needs.

- User-specified mode.
  - `--prompts`: SeeSR utilizes DAPE to automatically extract tag prompts from LR images, but it is not the most perfect approach. You can try manually specifying appropriate tag prompts to further enhance the quality of the results.

## 🌈 Train

#### Step1: Download the pretrained models
Download the pretrained [SD-2.1base models](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) and [RAM](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth). You can put them into `preset/models`.

#### Step2: Prepare training data
We pre-prepare training data pairs for the training process, which would take up some memory space but save training time. 

- For making paired data when training DAPE, you can run `utils_data/make_paired_data_DAPE.py`. 
- For making paired data when training SeeSR, you can run `utils_data/make_paired_data.py`

Please specify the `gt_path` at `line 35` before running these commands. If you have multi gt dirs, you can set it as ['PATH1', 'PATH2', 'PATH3', ...]. 

Once the degraded data pairs are created, you can base them to generate tag data by running `utils_data/make_tags.py`.

The data folder should be like this:
```
your_training_datasets/
    └── gt
        └── 0000001.png # GT images, (512, 512, 3)
        └── ...
    └── lr
        └── 0000001.png # LR images, (512, 512, 3)
        └── ...
    └── tag
        └── 0000001.txt # tag prompts
        └── ...
```

#### Step3: Training for DAPE
Please specify the DAPE training data path at `line 13` of `basicsr/options/dape.yaml`, then run the training command:
```
python basicsr/train.py -opt basicsr/options/dape.yaml
```
#### Step4: Traing for SeeSR
Please specify the SeeSR training data path at `--root_folders`; DAPE model path at `--ram_ft_path`; pretrained SD model path at `pretrained_model_name_or_path`. Then you can run the training command:

```
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7," accelerate launch train_seesr.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-base" \
--output_dir="./experience/seesr" \
--root_folders 'the path of your training datasets ' \
--ram_ft_path 'the path of your dape model from last step' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=512 \
--learning_rate=5e-5 \
--train_batch_size=2 \
--gradient_accumulation_steps=2 \
--null_text_ratio=0.5 
--dataloader_num_workers=0 \
--checkpointing_steps=10000 
```

The overall batch size is determined by num of `CUDA_VISIBLE_DEVICES`, `--train_batch_size`, and `--gradient_accumulation_steps` collectively. If your GPU memory is limited, you can consider reducing `--train_batch_size` while increasing `--gradient_accumulation_steps`.


## ❤️ Acknowledgments
This project is based on [diffusers](https://github.com/huggingface/diffusers) and [BasicSR](https://github.com/XPixelGroup/BasicSR). Some codes are brought from [PASD](https://github.com/yangxy/PASD) and [RAM](https://github.com/xinyu1205/recognize-anything). Thanks for their awesome works. We also pay tribute to the pioneering work of [StableSR](https://github.com/IceClear/StableSR).

## 📧 Contact
If you have any questions, please feel free to contact: `rong-yuan.wu@connect.polyu.hk`

## 🎓Citations
If our code helps your research or work, please consider citing our paper.
The following are BibTeX references:

```
@article{wu2023seesr,
  title={SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution},
  author={Wu, Rongyuan and Yang, Tao and Sun, Lingchen and Zhang, Zhengqiang and Li, Shuai and Zhang, Lei},
  journal={arXiv preprint arXiv:2311.16518},
  year={2023}
}
```

## 🎫 License
This project is released under the [Apache 2.0 license](LICENSE).

<details>
<summary>star history</summary>

[![Star History Chart](https://api.star-history.com/svg?repos=cswry/seesr&type=Date)](https://star-history.com/#cswry/seesr&Date)
</details>

<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=cswry/SeeSR)

</details>
