import gradio as gr
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import sys
from typing import List
# sys.path.append(os.getcwd())

import numpy as np
from PIL import Image

import torch
import torch.utils.checkpoint
from pytorch_lightning import seed_everything
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline

from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from torchvision import transforms
from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# Load scheduler, tokenizer and models.
pretrained_model_path = 'preset/models/stable-diffusion-2-1-base'
seesr_model_path = 'preset/models/seesr'

scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
feature_extractor = CLIPImageProcessor.from_pretrained(f"{pretrained_model_path}/feature_extractor")
unet = UNet2DConditionModel.from_pretrained(seesr_model_path, subfolder="unet")
controlnet = ControlNetModel.from_pretrained(seesr_model_path, subfolder="controlnet")

# Freeze vae and text_encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)
controlnet.requires_grad_(False)

if is_xformers_available():
    unet.enable_xformers_memory_efficient_attention()
    controlnet.enable_xformers_memory_efficient_attention()
else:
    raise ValueError("xformers is not available. Make sure it is installed correctly")

# Get the validation pipeline
validation_pipeline = StableDiffusionControlNetPipeline(
    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor,
    unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
)

validation_pipeline._init_tiled_vae(encoder_tile_size=1024,
                                    decoder_tile_size=224)
weight_dtype = torch.float16
device = "cuda"

# Move text_encode and vae to gpu and cast to weight_dtype
text_encoder.to(device, dtype=weight_dtype)
vae.to(device, dtype=weight_dtype)
unet.to(device, dtype=weight_dtype)
controlnet.to(device, dtype=weight_dtype)


tag_model = ram(pretrained='preset/models/ram_swin_large_14m.pth',
                pretrained_condition='preset/models/DAPE.pth',
                image_size=384,
                vit='swin_l')
tag_model.eval()
tag_model.to(device, dtype=weight_dtype)

@torch.no_grad()
def process(
    input_image: Image.Image,
    user_prompt: str,
    positive_prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    scale_factor: int,
    cfg_scale: float,
    seed: int,
    latent_tiled_size: int,
    latent_tiled_overlap: int,
    sample_times: int
    ) -> List[np.ndarray]:
    process_size = 512
    resize_preproc = transforms.Compose([
        transforms.Resize(process_size, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    # with torch.no_grad():
    seed_everything(seed)
    generator = torch.Generator(device=device)

    validation_prompt = ""
    lq = tensor_transforms(input_image).unsqueeze(0).to(device).half()
    lq = ram_transforms(lq)
    res = inference(lq, tag_model)
    ram_encoder_hidden_states = tag_model.generate_image_embeds(lq)
    validation_prompt = f"{res[0]}, {positive_prompt},"
    validation_prompt = validation_prompt if user_prompt=='' else f"{user_prompt}, {validation_prompt}"

    ori_width, ori_height = input_image.size
    resize_flag = False

    rscale = scale_factor
    input_image = input_image.resize((int(input_image.size[0] * rscale), int(input_image.size[1] * rscale)))

    if min(input_image.size) < process_size:
        input_image = resize_preproc(input_image)

    input_image = input_image.resize((input_image.size[0] // 8 * 8, input_image.size[1] // 8 * 8))
    width, height = input_image.size
    resize_flag = True  #

    images = []
    for _ in range(sample_times):
        try:
            image = validation_pipeline(
                validation_prompt, input_image, negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps, generator=generator,
                height=height, width=width,
                guidance_scale=cfg_scale,  conditioning_scale=1,
                start_point='lr', start_steps=999,ram_encoder_hidden_states=ram_encoder_hidden_states,
                latent_tiled_size=latent_tiled_size, latent_tiled_overlap=latent_tiled_overlap
            ).images[0]

            if True:  # alpha<1.0:
                image = wavelet_color_fix(image, input_image)

            if resize_flag:
                image = image.resize((ori_width * rscale, ori_height * rscale))
        except Exception as e:
            print(e)
            image = Image.new(mode="RGB", size=(512, 512))
        images.append(np.array(image))
    return images


#
MARKDOWN = \
"""
## SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution

[GitHub](https://github.com/cswry/SeeSR) | [Paper](https://arxiv.org/abs/2311.16518)

If SeeSR is helpful for you, please help star the GitHub Repo. Thanks!
"""

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", type="pil")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Options", open=True):
                user_prompt = gr.Textbox(label="User Prompt", value="")
                positive_prompt = gr.Textbox(label="Positive Prompt", value="clean, high-resolution, 8k, best quality, masterpiece")
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="dotted, noise, blur, lowres, oversmooth, longbody, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
                )
                cfg_scale = gr.Slider(label="Classifier Free Guidance Scale (Set a value larger than 1 to enable it!)", minimum=0.1, maximum=10.0, value=5.5, step=0.1)
                num_inference_steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=50, step=1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=231)
                sample_times = gr.Slider(label="Sample Times", minimum=1, maximum=10, step=1, value=1)
                latent_tiled_size = gr.Slider(label="Diffusion Tile Size", minimum=128, maximum=480, value=320, step=1)
                latent_tiled_overlap = gr.Slider(label="Diffusion Tile Overlap", minimum=4, maximum=16, value=4, step=1)
                scale_factor = gr.Number(label="SR Scale", value=4)
        with gr.Column():
            result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery").style(grid=2, height="auto")

    inputs = [
        input_image,
        user_prompt,
        positive_prompt,
        negative_prompt,
        num_inference_steps,
        scale_factor,
        cfg_scale,
        seed,
        latent_tiled_size,
        latent_tiled_overlap,
        sample_times,
    ]
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])

block.launch()

