"""A randomly initialised, tiny instance of the REAL upstream SeeSR pipeline for CPU tests.

Uses the upstream classes (models/unet_2d_condition.py, models/controlnet.py,
pipelines/pipeline_seesr.py) with diffusers' AutoencoderKL and DDPMScheduler configured like
SD-2-base. Only the sizes are small; the sampling code path, CFG, LR-embedded start, tiled-VAE
hook and every RNG call are the ones used in real inference. No weights are downloaded.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SD2_BASE_SCHEDULER = {  # stable-diffusion-2-base/scheduler/scheduler_config.json as published
    "_class_name": "PNDMScheduler", "beta_end": 0.012, "beta_schedule": "scaled_linear", "beta_start": 0.00085,
    "clip_sample": False, "num_train_timesteps": 1000, "prediction_type": "epsilon", "set_alpha_to_one": False,
    "skip_prk_steps": True, "steps_offset": 1, "trained_betas": None,
}


def build_pipeline(tmp_dir):
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import CLIPImageProcessor, CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel
    from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4, layers_per_block=1, block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32, attention_head_dim=8, use_image_cross_attention=True)
    controlnet = ControlNetModel.from_unet(unet, use_image_cross_attention=True)
    # ControlNet's zero convolutions start at exactly zero (as in training), which would make every
    # residual zero and any test of LR conditioning vacuous. Give them small random values instead.
    with torch.no_grad():
        for prm in controlnet.parameters():
            if not prm.any():
                prm.normal_(0.0, 0.05)
    vae = AutoencoderKL(block_out_channels=(32, 32, 32, 32), in_channels=3, out_channels=3, latent_channels=4,
                        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                        layers_per_block=1)
    text_encoder = CLIPTextModel(CLIPTextConfig(hidden_size=32, intermediate_size=37, num_hidden_layers=2,
                                                num_attention_heads=4, vocab_size=100, max_position_embeddings=77))
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for c in "abcdefghijklmnopqrstuvwxyz,":
        vocab[c] = len(vocab)
        vocab[c + "</w>"] = len(vocab)
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "vocab.json").write_text(json.dumps(vocab))
    (tmp_dir / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(tmp_dir / "vocab.json"), str(tmp_dir / "merges.txt"), model_max_length=77)
    pipe = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet,
        scheduler=DDPMScheduler.from_config(SD2_BASE_SCHEDULER), safety_checker=None,
        feature_extractor=CLIPImageProcessor(), requires_safety_checker=False)
    pipe._init_tiled_vae(encoder_tile_size=1024, decoder_tile_size=224)  # as test_seesr.load_seesr_pipeline
    pipe.set_progress_bar_config(disable=True)
    return pipe


def call_kwargs(image, steps, generator, ram_embeds):
    """The keyword arguments test_seesr.main() passes (upstream defaults), at test size."""
    return dict(num_inference_steps=steps, generator=generator, height=image.height, width=image.width,
                guidance_scale=5.5, negative_prompt="dotted, noise, blur, lowres, smooth", conditioning_scale=1.0,
                start_point="lr", ram_encoder_hidden_states=ram_embeds, latent_tiled_size=96,
                latent_tiled_overlap=4, args=object(), output_type="np")
