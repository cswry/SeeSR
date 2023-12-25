# single gpu
CUDA_VISIBLE_DEVICES="0," accelerate launch train_seesr.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-base" \
--output_dir="./experience/seesr" \
--root_folders 'preset/datasets/training_datasets' \
--ram_ft_path 'preset/models/DAPE.pth' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=512 \
--learning_rate=5e-5 \
--train_batch_size=16 \
--gradient_accumulation_steps=2 \
--null_text_ratio=0.5 
--dataloader_num_workers=0 \
--checkpointing_steps=10000 


# multi-gpus
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7," accelerate launch train_seesr.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-2-base" \
--output_dir="./experience/seesr" \
--root_folders 'preset/datasets/training_datasets' \
--ram_ft_path 'preset/models/DAPE.pth' \
--enable_xformers_memory_efficient_attention \
--mixed_precision="fp16" \
--resolution=512 \
--learning_rate=5e-5 \
--train_batch_size=2 \
--gradient_accumulation_steps=2 \
--null_text_ratio=0.5 
--dataloader_num_workers=0 \
--checkpointing_steps=10000 