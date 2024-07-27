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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf 
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np

def readImages_RAW(renders_dir1, renders_dir2, renders_dir3, renders_raw_dir, gt_dir, gt_raw_dir):
    renders1, renders2, renders3, renders_raw, = [], [], [], []
    gts, gts_raw = [], []
    image_names = []
 
    for fname in os.listdir(renders_dir1):
        # print(fname)
        render1 = Image.open(renders_dir1 / fname)
        render2 = Image.open(renders_dir2 / fname)
        render3 = Image.open(renders_dir3 / fname)
        render_raw = np.load(renders_raw_dir / (fname.split(".")[0]+".npy"))
        gt = Image.open(gt_dir / fname)
        gt_raw = np.load(gt_raw_dir / (fname.split(".")[0]+".npy"))
        if render1.size != gt.size:
            new_size = gt.size
            render1 = render1.resize(new_size)
            render2 = render2.resize(new_size)
            render3 = render3.resize(new_size)
          

        renders1.append(tf.to_tensor(render1).unsqueeze(0)[:, :3, :, :].cuda())
        renders2.append(tf.to_tensor(render2).unsqueeze(0)[:, :3, :, :].cuda())
        renders3.append(tf.to_tensor(render3).unsqueeze(0)[:, :3, :, :].cuda())
        renders_raw.append(torch.tensor(render_raw).unsqueeze(0).cuda())       
        
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        gts_raw.append(torch.tensor(gt_raw).unsqueeze(0).cuda())
      
        image_names.append(fname)
 
    return renders1, renders2, renders3, renders_raw, gts, gts_raw, image_names

def readImages(renders_dir1, gt_dir):
    renders1, gts = [], []
    image_names = []
 
    for fname in os.listdir(renders_dir1):
        # print(fname)
        render1 = Image.open(renders_dir1 / fname)
      
        gt = Image.open(gt_dir / fname)
      

        renders1.append(tf.to_tensor(render1).unsqueeze(0)[:, :3, :, :].cuda())
      
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
   
      
        image_names.append(fname)
 
    return renders1, gts, image_names


def evaluate(model_paths,scene_path, is_raw, do_train,eval_mode="HDR"):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        # try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}
            if not do_train:
                test_dir = Path(scene_dir) / "test"
            else:
                test_dir = Path(scene_dir) / "train"
    
            print(test_dir)
            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                if eval_mode == "HDR":
                    gt_dir = Path(scene_dir) / "gt_hdr"
                else:
                    gt_dir = method_dir/ "gt"
                renders_dir1 = method_dir / "renders"
                if is_raw:
                    gt_raw_dir = method_dir / "gt_raw"
                    renders_dir2 = method_dir / "renders_affine"
                    renders_dir3 = method_dir / "renders_cc"
                    renders_raw_dir = method_dir / "renders_raw"
             
                if is_raw:
                    renders1, renders2, renders3, renders_raw,  gts, gts_raw, image_names = readImages_RAW(renders_dir1,renders_dir2,renders_dir3, renders_raw_dir, gt_dir, gt_raw_dir)
                else:
                    renders1, gts, image_names = readImages(renders_dir1, gt_dir)
                ssims = []
                psnrs = []
                lpipss = []

          

                for idx in tqdm(range(len(renders1)), desc="Metric evaluation progress (postprocess)"):
                    ssims.append(ssim(renders1[idx], gts[idx]))
                    psnrs.append(psnr(renders1[idx], gts[idx]))
                    lpipss.append(lpips(renders1[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")
                full_dict[scene_dir][method].update({"SSIM1": torch.tensor(ssims).mean().item(),
                                                        "PSNR1": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS1": torch.tensor(lpipss).mean().item()})
                
                per_view_dict[scene_dir][method].update({"PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)}})
                if is_raw:
                    ssims = []
                    psnrs = []
                    lpipss = []
                    
                    for idx in tqdm(range(len(renders2)), desc="Metric evaluation progress (affine color transform)"):
                        ssims.append(ssim(renders2[idx], gts[idx]))
                        psnrs.append(psnr(renders2[idx], gts[idx]))
                        lpipss.append(lpips(renders2[idx], gts[idx], net_type='vgg'))

                    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                    print("")
                    full_dict[scene_dir][method].update({"SSIM2": torch.tensor(ssims).mean().item(),
                                                            "PSNR2": torch.tensor(psnrs).mean().item(),
                                                            "LPIPS2": torch.tensor(lpipss).mean().item()})
                    
                    ssims = []
                    psnrs = []
                    lpipss = []
                    
                    for idx in tqdm(range(len(renders3)), desc="Metric evaluation progress (color correctrion)"):
                        ssims.append(ssim(renders3[idx], gts[idx]))
                        psnrs.append(psnr(renders3[idx], gts[idx]))
                        lpipss.append(lpips(renders3[idx], gts[idx], net_type='vgg'))

                    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                    print("")
                    full_dict[scene_dir][method].update({"SSIM3": torch.tensor(ssims).mean().item(),
                                                            "PSNR3": torch.tensor(psnrs).mean().item(),
                                                            "LPIPS3": torch.tensor(lpipss).mean().item()})
                    
                    
                
                    psnrs = []
                    
                    for idx in tqdm(range(len(renders_raw)), desc="Metric evaluation progress (RAW)"):
                        psnrs.append(psnr(renders_raw[idx], gts_raw[idx]))

                    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                 
                    print("")
                    full_dict[scene_dir][method].update({"PSNR_RAW": torch.tensor(psnrs).mean().item()})
                                                           

             

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
      

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument("--is_raw", "-r", action="store_true")
    parser.add_argument('--train', '-t', action="store_true")
    parser.add_argument('--eval_mode', '-e', default="LDR")
    parser.add_argument('--scene_path', '-s', default="/media/cilab/data/shreyas/RAWHDR_dataset/hostelroom")
    args = parser.parse_args()
    do_train = args.train
    is_raw  = args.is_raw 
    eval_mdoe = args.eval_mode
    evaluate(args.model_paths, args.scene_path, is_raw , do_train,eval_mdoe) 

