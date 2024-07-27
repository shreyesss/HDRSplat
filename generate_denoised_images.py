#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.raw_utils import bilinear_demosaic_np
from utils.multinerf_utils import open_file
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.raw_utils import load_raw_dataset_from_scratch
from raw_enhancement.PMRID.models.inference import denoise
import matplotlib.pyplot as plt 
from internal import image as lib_image
import numpy as np
from PIL import Image
import bm3d
import cv2
import jax 
import jax.numpy as jnp
import rawpy
import json
import pickle
import time 
# Denoising (PMRID) the RAW images as a Pre Processing step and storing them
def denoise_set_raw(scene_path):

    # Demosaiced RAW images 
    gts_path_post_proc_raw = os.path.join(scene_path, "denoised","demosaiced_raw")
    gts_path_post_proc = os.path.join(scene_path, "denoised","demosaiced")
    # Demosaiced+Denoised (PMRID) RAW images
    denoised_path_raw1 = os.path.join(scene_path, "denoised", "PMRID_raw")
    denoised_path_1 = os.path.join(scene_path, "denoised", "PMRID")
    metadata_path = os.path.join(scene_path,"denoised", "metadata")
   

    # Generate paths for storing the denoisaied & denoised RAW images 
    makedirs(gts_path_post_proc, exist_ok=True)
    makedirs(gts_path_post_proc_raw, exist_ok=True)
    makedirs(denoised_path_raw1, exist_ok=True)
    makedirs(denoised_path_1, exist_ok=True)
    makedirs(metadata_path, exist_ok=True)
   

    for idx, img_name in enumerate(os.listdir(os.path.join(scene_path,"images"))):
            print(f"Processing {idx+1} {img_name}")
            save_ext = "."+img_name.split(".")[1]
            img_name = img_name.split(".")[0]
            raw_folder = os.path.join(scene_path, "raw")
            t1 = time.time()
            # Load the bayer RAW image from the dataset and the corresponding meta-data
            raw, demosaiced_raw, meta, post_fn = load_raw_dataset_from_scratch(scene_path=scene_path,raw_folder=raw_folder,image_name=img_name)
            print("loading",time.time() - t1)
            # Demosaic the RAW image
            demosaiced_raw2rgb = post_fn(demosaiced_raw)
            t1 = time.time()
            # Denoise the RAW image (PMRID)
            denoised_bayer = denoise(raw[0],meta["ISO"])
            print("denoising",time.time() - t1)
            demosaiced_denoised_raw = bilinear_demosaic_np(denoised_bayer)
            denoised_raw2rgb = post_fn(demosaiced_denoised_raw)
           
            # Save the meta-data to disk
            dict_save_path = os.path.join(metadata_path, img_name.split(".")[0] + ".pkl")
            with open(dict_save_path, 'wb') as f:
                pickle.dump(meta, f)
          

            # Save the the demosaiced and denoised images to disk          
            np.save(os.path.join(denoised_path_raw1, img_name + ".npy"), demosaiced_denoised_raw)
            plt.imsave(os.path.join(denoised_path_1, img_name + save_ext),(255*denoised_raw2rgb).astype(np.uint8))
            np.save(os.path.join(gts_path_post_proc_raw, img_name.split(".")[0] + ".npy"), demosaiced_raw)
            plt.imsave(os.path.join(gts_path_post_proc, img_name + save_ext),(255*demosaiced_raw2rgb).astype(np.uint8))
           
         

        
        
    

if __name__ == "__main__":
  
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--scene_path", type = str, default = "true")
    args = parser.parse_args()
    denoise_set_raw(args.scene_path)
  
