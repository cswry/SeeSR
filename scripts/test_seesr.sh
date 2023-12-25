python test_seesr.py \
--pretrained_model_path preset/models/stable-diffusion-2-base \
--prompt None \
--seesr_model_path preset/models/seesr \
--ram_ft_path preset/models/DAPE.pth \
--image_path preset/datasets/test_datasets \
--output_dir preset/datasets/output \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--process_size 512 

