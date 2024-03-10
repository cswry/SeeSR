Default inference settings
```
python test_seesr.py \
--pretrained_model_path preset/models/stable-diffusion-2-base \
--prompt '' \
--seesr_model_path preset/models/seesr \
--ram_ft_path preset/models/DAPE.pth \
--image_path preset/datasets/test_datasets \
--output_dir preset/datasets/output \
--start_point lr \
--num_inference_steps 50 \
--guidance_scale 5.5 \
--process_size 512 
```

The default settings are optimized for the best result. However, the behavior of the SeeSR can be customized
- Trade-offs between the **fidelity** and **perception**  
  - `--num_inference_steps` Using more sampling steps in `Real-world SR` tasks is not a purely beneficial choice. While it improves the perception quality, it can also reduce fidelity quality as it generates more. Considering the trade-offs between fidelity and perception, as well as the inference time cost, we set the default value to `50`. However, you can make appropriate adjustments based on your specific needs.
  - `--guidance_scale` A higher value means unleashing more generation capacity of SD, which improves perception quality but decreases fidelity quality. We set the default value to `5.5`, you can make appropriate adjustments based on your specific needs.
  - `--process_size` The inference script resizes input images to the `process_size`, and then resizes the prediction back to the original resolution after process. We found that increasing the processing size (e.g. 768) improves fidelity but decreases perception. We set the default value to `512`, consistent with the training size of the pre-trained SD model. You can make appropriate adjustments based on your specific needs.

- User-specified mode
  - `--prompt` SeeSR utilizes DAPE to automatically extract tag prompts from LR images, but it is not the most perfect approach. You can try manually specifying appropriate tag prompts to further enhance the quality of the results.