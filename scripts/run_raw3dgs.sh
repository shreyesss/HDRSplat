#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

folder_name="candlefiat"
scene_path="/media/cilab/data/shreyas/rawnerf-undistorted/scenes/${folder_name}"
output_path="/media/cilab/data/shreyas/Results/temp/${folder_name}"

# Execute training
python train.py  -s "$scene_path" --model_path "$output_path" --resolution 4  --densify_grad_thresh 0.00012 --densify_until_iter 10000 \
                --iterations 30000 --test_iterations 7000 30000 --save_iterations 7000 30000  --position_lr_init 0.00016 \
                --scaling_lr 0.005 --opacity_lr 0.005 --feature_lr 0.0025 --denoise_method "demosaic" --loss_mode "raw3dgs" \
                --render_iterations 7000 30000 --checkpoint_iterations 30000  --sh_degree 3 --port 1216 --eval --is_raw --add_points
                

python metrics.py -m "$output_path" --is_raw


   
