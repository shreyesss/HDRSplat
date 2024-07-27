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

import cv2
import torch
import trimesh
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
# import matplotlib.pyplot as plt 
from internal.image import color_correct
from internal.raw_utils import match_images_affine
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.graphics_utils import getWorld2View2, focal2fov, getIntrinsicMatrix
import numpy as np
from PIL import Image

import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "2ours_{}".format(iteration), "renders")
    gts_raw_path = os.path.join(model_path, name, "2ours_{}".format(iteration), "gt_postprocess_raw")
    gts_path = os.path.join(model_path, name, "2ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gts_raw_path, exist_ok=True)
    # print(views)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        rendering = torch.permute(rendering,(1,2,0))
        rendering = rendering.detach().cpu().numpy()
        plt.imsave(os.path.join(render_path, '{}'.format(view.image_name) + ".png"),(255*rendering).astype(np.uint8))
        
        gt_image  = view.original_image[0:3, :, :]
    
        gt_image = torch.permute(gt_image,(1,2,0))
        gt_image = gt_image.detach().cpu().numpy()

        plt.imsave(os.path.join(gts_path, '{}'.format(view.image_name) + ".png"),(255*gt_image).astype(np.uint8))

def render_set_raw(model_path, scene_path, name, iteration, views, gaussians, pipeline, background, is_affine_cc, is_cc):
    render_path_affine = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_affine")
    render_path_color_correct = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_cc")
    render_raw_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_raw")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_raw_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_raw")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gts_path_post_proc = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_postprocess")

    makedirs(render_path, exist_ok=True)
    makedirs(render_path_affine, exist_ok=True)
    makedirs(render_path_color_correct, exist_ok=True)
    makedirs(render_raw_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gts_raw_path, exist_ok=True)
    makedirs(gts_path_post_proc, exist_ok=True)
 
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # t1 = time.time()
        render_pkg = render(view, gaussians, pipeline, background)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # print("render", time.time() - t1)
        resolution = (view.original_image_raw.shape[2], view.original_image_raw.shape[1])
        # Hardcoded extension
        # print(viewspace_point_tensor[10000:10005, ...])
        # print(visibility_filter[:5])
        # print(radii[:5])
       
        extensions = [".jpg",".JPG"] 

        for ext in extensions:
            image_path = os.path.join(scene_path, "images", view.image_name + ext)
            # print(image_path)
            if os.path.exists(image_path):
                gt_image = Image.open(image_path).resize(resolution) 
                gt_image = np.array(gt_image) / 255.       
            else:
                continue 

        

        gt_raw = view.original_image_raw
        gt_raw = torch.permute(gt_raw,(1,2,0))
        gt_raw = gt_raw.detach().cpu().numpy()
        # print(gt_raw.min(), gt_raw.max())
        #postprocess raw image
        # gt_raw2rgb = view.meta(gt_raw) 

        # plt.imsave(os.path.join(gts_raw_path, '{0:05d}'.format(idx) + ".png"),(255*gt_raw).astype(np.uint8))
        np.save(os.path.join(gts_raw_path, '{}'.format(view.image_name) + ".npy"), gt_raw)
        plt.imsave(os.path.join(gts_path, '{}'.format(view.image_name) + ".png"),(255 * gt_image).astype(np.uint8))


        rendering = torch.permute(rendering,(1,2,0))
        rendering = rendering.detach().cpu().numpy()
        # print(rendering.min(), rendering.max())
        np.save(os.path.join(render_raw_path, '{}'.format(view.image_name) + ".npy"), rendering)

        # affine align in raw space b/w gt_raw and model output and then post process
        # aligned_raw = match_images_affine(rendering, gt_raw)
        # render_aligned_raw = view.meta(aligned_raw)
        # plt.imsave(os.path.join(render_path_raw_align, '{0:05d}'.format(idx) + ".png"),(255*render_aligned_raw).astype(np.uint8))

        # postprocessing raw_gt for comparison
        # t1 = time.time()
        raw_rgb = view.post_fn(gt_raw)
        # print("pp", time.time() -t1)
        plt.imsave(os.path.join(gts_path_post_proc, '{}'.format(view.image_name) + ".png"),(255*raw_rgb).astype(np.uint8))
        # postprocessing using meta data 
        rendering_rgb = view.post_fn(rendering)
        plt.imsave(os.path.join(render_path, '{}'.format(view.image_name) + ".png"),(255*rendering_rgb).astype(np.uint8))

        # affine color alignment 
        if is_affine_cc:
            rendering_rgb_cc = match_images_affine(rendering_rgb, gt_image)
            plt.imsave(os.path.join(render_path_affine, '{}'.format(view.image_name) + ".png"),(255*rendering_rgb_cc).astype(np.uint8))
        
        # color correction
        if is_cc:
            # t1 = time.time()
            rendering_rgb_cc = color_correct(rendering_rgb, gt_image,5,0.)
            # print("cc", time.time() -t1)
            plt.imsave(os.path.join(render_path_color_correct, '{}'.format(view.image_name) + ".png"),(255*rendering_rgb_cc).astype(np.uint8))
        
        # # sys.exit()
       
       
def project_point_cloud(model_path,scene_path,iteration):

    def load_camera(scene_path):
        znear = 0.1
        zfar = 1000
    
        cameras_extrinsic_file = os.path.join(scene_path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(scene_path, "sparse/0", "cameras.bin")
    
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    

        key = 10
        extr = cam_extrinsics[key]

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        t = np.array(extr.tvec).reshape(3,1)
        K = getIntrinsicMatrix(intr.params)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

        return R, t, K, height, width, extr.name, extr.xys 

    R, t, K, h, w, image_name, xys = load_camera(scene_path)
   

    def get_2D_points(mesh,K,T,h,w):

        Pw = mesh.vertices # [N,3]
        Pw = np.hstack((Pw, np.ones((len(Pw), 1)))).T # [4,N]
        Pc = np.dot(T, Pw).T # [N,3]
        Pc_norm = (Pc / Pc[:, 2][:, np.newaxis]).T #[N,3]
        uv = np.dot(K, Pc_norm).T
     
        uv_new = uv[(uv[:, 0] >= 0) & (uv[:, 0] <= w) &
                                    (uv[:, 1] >= 0) & (uv[:, 1] <= h)]
        return uv_new[:,0].astype(np.uint32), uv_new[:,1].astype(np.uint32)

  
 
    final_mesh_path = os.path.join(model_path,"point_cloud","iteration_{}".format(iteration), "point_cloud.ply")
    initial_mesh_path = os.path.join(model_path, "input.ply")
    mesh1 = trimesh.load_mesh(initial_mesh_path)
    mesh2 = trimesh.load_mesh(final_mesh_path)
    image_path = os.path.join(scene_path, "images", image_name)

   
    image = cv2.imread(image_path)
    image1 = cv2.resize(image,(w,h))
    image2 =  cv2.resize(image,(w,h))
    image3 =  cv2.resize(image,(w,h))
    # print(image.shape,h,w)
   
   
    T = getWorld2View2(R,t[:,0])[:3,:]
    x1, y1 = get_2D_points(mesh1,K,T,h,w)
    x2, y2 = get_2D_points(mesh2,K,T,h,w)

    for (x,y) in zip(x1,y1):
        cv2.circle(image1, (x,y), 2, (255, 255, 0), -1)
    cv2.imwrite(os.path.join(model_path,"pointcloud-intial.png"), image1)

    for (x,y) in zip(x2,y2):
        cv2.circle(image2, (x,y), 2, (0, 255, 255), -1)
    cv2.imwrite(os.path.join(model_path,f"pointcloud-final-{iteration}.png"), image2)

    for (x,y) in xys:
        cv2.circle(image3, (int(x),int(y)), 2, (255, 0, 255), -1)
    cv2.imwrite(os.path.join(model_path,"SfM-features-intial.png"), image3)
      


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
   
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        is_cc = dataset.color_correct
        is_affine_cc = dataset.affine_color_fit
        if not skip_train:
            if dataset.is_raw:
                render_set_raw(dataset.model_path,dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, is_affine_cc, is_cc)
                # render_set_raw(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, is_affine_cc, is_cc)
            else:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #     if dataset.is_raw:
        #         render_set_raw(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,is_affine_cc, is_cc)
        #     else:
        #         render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30_000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")



  
   
 

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), 30000, pipeline.extract(args), args.skip_train, args.skip_test)
