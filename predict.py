# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
from cog import BasePredictor, Input, Path
import os
import time
import subprocess
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.utils.checkpoint
from pytorch_lightning import seed_everything
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline

from utils.wavelet_color_fix import wavelet_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from torchvision import transforms
from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel

MODEL_URL = "https://weights.replicate.delivery/default/stabilityai/sd-2-1-base.tar"

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
device = "cuda"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Load scheduler, tokenizer and models.
        pretrained_model_path = 'preset/models/stable-diffusion-2-1-base'
        seesr_model_path = 'preset/models/seesr'

        # Download SD-2-1 weights
        if not os.path.exists(pretrained_model_path):
            download_weights(MODEL_URL, pretrained_model_path)

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
        validation_pipeline._init_tiled_vae(encoder_tile_size=1024,decoder_tile_size=224)
        self.validation_pipeline = validation_pipeline
        weight_dtype = torch.float16
    
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
        self.tag_model = tag_model.to(device, dtype=weight_dtype)


    # @torch.no_grad()
    def process(
        self,
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

        seed_everything(seed)
        generator = torch.Generator(device=device)

        validation_prompt = ""
        lq = tensor_transforms(input_image).unsqueeze(0).to(device).half()
        lq = ram_transforms(lq)
        res = inference(lq, self.tag_model)
        ram_encoder_hidden_states = self.tag_model.generate_image_embeds(lq)
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
        resize_flag = True

        images = []
        for _ in range(sample_times):
            try:
                with torch.autocast("cuda"):
                    image = self.validation_pipeline(
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
    

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image"),
        user_prompt: str = Input(description="Prompt to condition on", default=""),
        positive_prompt: str = Input(description="Prompt to add", default="clean, high-resolution, 8k"),
        negative_prompt: str = Input(description="Prompt to remove", default="dotted, noise, blur, lowres, smooth"),
        cfg_scale: float = Input(description="Guidance scale, set value to >1 to use", default=5.5, ge=0.1, le=10.0),
        num_inference_steps: int = Input(description="Number of inference steps", default=50, ge=10, le=100),
        sample_times: int = Input(description="Number of samples to generate", default=1, ge=1, le=10),
        latent_tiled_size: int = Input(description="Size of latent tiles", default=320, ge=128, le=480),
        latent_tiled_overlap: int = Input(description="Overlap of latent tiles", default=4, ge=4, le=16),
        scale_factor: int = Input(description="Scale factor", default=4),
        seed: int = Input(description="Seed", default=231, ge=0, le=2147483647),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        pil_image = Image.open(image).convert("RGB")
        imgs = self.process(
            pil_image, user_prompt, positive_prompt, negative_prompt, num_inference_steps,
            scale_factor, cfg_scale, seed, latent_tiled_size, latent_tiled_overlap, sample_times)

        # Clear output folder
        os.system("rm -rf /tmp/output")
        # Create output folder
        os.system("mkdir /tmp/output")
        # Save images to output folder
        output_paths = []
        for i, img in enumerate(imgs):
            img = Image.fromarray(img)
            output_path = f"/tmp/output/{i}.png"
            img.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths