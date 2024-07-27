#!/bin/bash

folder_name="candlefiat"
scene_path="/media/cilab/data/shreyas/rawnerf-undistorted/scenes/${folder_name}"
output_path="/media/cilab/data/shreyas/Results/temp/${folder_name}"

# Execute training

python train.py  -s "$scene_path" --model_path "$output_path" --resolution 16 \
                --iterations 30000 --test_iterations 7000 30000 --save_iterations 7000 30000  --position_lr_init 0.00008 \
                --scaling_lr 0.001 --opacity_lr 0.005 --feature_lr 0.0005 --denoise_method "demosaic" --loss_mode "hdrsplat" \
                --render_iterations 7000 30000 --checkpoint_iterations 30000  --sh_degree 3 --port 1219 --eval --is_raw 

                
        
# python render.py -m "$output_path" --is_raw
python metrics.py -m "$output_path" --is_raw 

   
