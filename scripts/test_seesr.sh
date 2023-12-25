python test_seesr.py \
--pretrained_model_path /home/notebook/data/group/LowLevelLLM/models/diffusion_models/stable-diffusion-2-base \
--seesr_model_path preset/SeeSR_1225/seesr \
--ram_ft_path preset/models/DAPE.pth \
--image_path preset/datasets/test_datasets \
--output_dir preset/datasets/output_1 \
--num_inference_steps 50 \
--start_point lr \
--start_steps 999 \
--process_size 512 \
--upscale 1 \
--guidance_scale 5.5

