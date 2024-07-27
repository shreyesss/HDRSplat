#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

folder_name=$(basename "$folder")

echo "Processing folder: $folder_name"

folder_name="candlefiat"
scene_path="/media/cilab/data/shreyas/rawnerf-undistorted/scenes/${folder_name}"
output_path="/media/cilab/data/shreyas/Results/temp/${folder_name}"

# Execute training
python train.py  -s "$scene_path" --model_path "$output_path" --resolution 4  --densify_grad_thresh 0.0002 --densify_until_iter 15000 \
                --iterations 30000 --test_iterations 7000 30000 --save_iterations 7000 30000 \
                --render_iterations 7000 30000  --port 1416 --eval 
            
            

python metrics.py -m "$output_path"

 