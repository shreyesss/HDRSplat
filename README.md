# [BMVC 2024] HDRSplat: Gaussian Splatting for High Dynamic Range 3D Scene Reconstruction from Raw Images
Shreyas Singh*, Aryan Garg*, Kaushik Mitra (* indicates equal contribution)<br>
[Webpage](https://aryan-garg.github.io/hdrsplat/) | [arXiv](https://arxiv.org/abs/2407.16503)

![Alt text](https://github.com/shreyesss/HDRSplat-3DGS-for-HDR-scene-reconstruction/blob/main/assets/main.jpg)
![Alt text](https://github.com/shreyesss/HDRSplat-3DGS-for-HDR-scene-reconstruction/blob/main/assets/pointcloud.jpg)



Abstract: *The recent advent of 3D Gaussian Splatting (3DGS) has revolutionized the 3D scene reconstruction space enabling high-fidelity novel view synthesis in real-time. However, with the exception of RawNeRF, all prior 3DGS and NeRF-based methods rely on 8-bit tone-mapped Low Dynamic Range (LDR) images for scene reconstruction. Such methods struggle to achieve accurate reconstructions in scenes that require a higher dynamic range. Examples include scenes captured in nighttime or poorly lit indoor spaces having a low signal-to-noise ratio, as well as daylight scenes with shadow regions exhibiting extreme contrast. Our proposed method HDRSplat tailors 3DGS to train directly on 14-bit linear raw images in near darkness which preserves the scenes' full dynamic range and content. Our key contributions are two-fold: Firstly, we propose a linear HDR space-suited loss that effectively extracts scene information from noisy dark regions and nearly saturated bright regions simultaneously, while also handling view-dependent colors without increasing the degree of spherical harmonics. Secondly, through careful rasterization tuning, we implicitly overcome the heavy reliance and sensitivity of 3DGS on point cloud initialization. This is critical for accurate reconstruction in regions of low texture, high depth of field, and low illumination. HDRSplat is the fastest method to date that does 14-bit (HDR) 3D scene reconstruction in ≤15 minutes/scene (∼30x faster than prior state-of-the-art RawNeRF). It also boasts the fastest inference speed at ≥120fps. We further demonstrate the applicability of our HDR scene reconstruction by showcasing various applications like synthetic defocus, dense depth map extraction, and post-capture control of exposure, tone-mapping and view-point.*


<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{singh24_hdrsplat,
  author    = {Singh, Shreyas and Garg, Aryan and Mitra, Kaushik},
  title     = {HDRSplat: Gaussian Splatting for High Dynamic Range 3D Scene Reconstruction from Raw Images},
  journal   = {BMVC},
  year      = {2024},
}
}</code></pre>
  </div>
</section>




## Cloning the Repository


```shell
# SSH
git clone git@github.com:graphdeco-inria/gaussian-splatting.git
```
or
```shell
# HTTPS
git clone https://github.com/graphdeco-inria/gaussian-splatting
```

## Overview

The codebase has 3 main components:
- 3D gaussian splatitng based differentiable rasterization for 3D scene reconstruction form Raw images
- A PMRID based pre-processing and denoising step in the Bayer-Raw space
- A flexible Image Signal Processing Pipeline to convert 14-bit Raw images to 8 bit images for display

![Alt text](https://github.com/shreyesss/HDRSplat-3DGS-for-HDR-scene-reconstruction/blob/main/assets/architecture.jpg)



### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)
- Please see FAQ for smaller VRAM configurations

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used Visual Studio 2019 for Windows)
- CUDA SDK 11 for PyTorch extensions, install *after* Visual Studio (we used 11.8, **known issues with 11.6**)
- C++ Compiler and CUDA SDK must be compatible

### Setup
The code has been developed on Ubuntu 20.02
Please note that this process assumes that you have CUDA SDK **11** installed, not **12**.

#### Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```

To set-up the **differentiable rasterization** and **simple-knn** modules run
```shell
cd submodules/simple-knn
pip install -e .
cd ../diff-gaussian-rasterization
pip install -e .
```
## Dataset

HDRSplat uses the RawNeRF dataset introduced by [Mildenhall.et.al](https://bmild.github.io/rawnerf/). The dataset can be downloaded from their website. The directory structure should look like this.
```
<location>
|---scene1
      |---raw
      |   |---<raw image 0>
      |   |---<raw image metadata 0>
      |   |---<raw image 1>
      |   |---<raw image metadata 1>
      |   |---...
      |---images
      |   |---<image 0>
      |   |---<image 1>
      |   |---...
      |---sparse
          |---0
              |---cameras.bin
              |---images.bin
              |---points3D.bin
|---scene2
|---....
```
**1. To proccess your own dataset, or generate a HDRSplat style dataset from RawNeRF dataset follow these steps:**
To create a train test split for a scene using random sampling, run
```shell
python create_train_test_split.py <path to COLMAP dataset scene> --test_percentage 15
```
To demosaic and then denoise the images using PMRID and save them to disk, run
```shell
python generate_denoised_images.py --scene_path <path to COLMAP dataset>
```
this will create a sub-folder called **denoised** in the scene folder with the following structure. 
```
<location>
|---raw
|---images
|---sparse
|---denoised
      |---PMRID_raw
      |      |---<Demosaiced & Denoised raw image 0>
      |      |---<Demosaiced & Denoised raw image 1>
      |      |---...
      |---PMRID
      |      |---<Demosaiced & Denoised raw image converted to 8 bit LDR 0>
      |      |---<Demosaiced & Denoised raw image converted to 8 bit LDR 1>
      |      |---...
      |---demosaiced_raw
      |      |---<Demosaiced raw image 0>
      |      |---<Demosaiced  raw image 1>
      |      |---...
      |---demosaiced
      |      |---<Demosaiced raw image converted to 8 bit LDR 0>
      |      |---<Demosaiced raw image converted to 8 bit LDR 1>
      |      |---...
      |---metadata
      |      |---<metadata for raw image 0>
      |      |---<metadata for raw image 1>
      |      |---...
```
The follwing folder contains the Demosaied Bayer raw images and PMRID denoised images and their corresponding LDR versions (converted using our minimalistic pipeline for visualization). The **PMRID denoised images** are used to train our **HDRSplat** model and the **simply demosaiced** images are used to train our RAw3DGS baseline. The script additionally also generates a metadata.ply file for each view in the scene. The follwing 5 metadata values are essential for our end to end pipeline:
1. ISO, 
2. Exposure
3. BlackLevel
4. WhiteLevel
5. Cam2RGB

**Or**

**2. Directly download our processed dataset from [Link](DummyLink), and skip straight to the training and evaluation stage!**

## Training & Evaluation

To train our HDRSplat model
```shell
bash ./scripts/run_hdrsplat.sh
```
To train the RAw3DGS baseline model (3DGS trained directly with RawNeRF loss, w/o PMRID denoising), run
```shell
bash ./scripts/run_raw3DGS.sh
```
To train the LDR3DGS baseline model (3DGS directly trained on 8-bit LDR images), run
```shell
bash ./scripts/run_ldr3DGS.sh
```
**_NOTE_**: Before running the scripts please edit the follwing variables in the script:
1. folder_name: [name of the scene]
2. scene_path: [path to the processed raw dataset]
3. output_path: [path to store the generated checkpoints and results>]

**_NOTE_** The script trains renders and evaluates the models all in one go!

To independently render a checkpoint for a scene, run 
```shell
python render.py -m <path to trained model> # Generate renderings
```
To independently evalauate a checkpointfor a scene, run
```shell
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```


<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>
<br>

## Benchmarking (Coming Soon)
To render and evalaute the models presented by us in the paper for benchmarking:
1. Download our trained checkpoints:

    i.  [HDRSplat](Link1)  
    ii.  [Raw3DGS (Baseline)](Link2)  
    iii.  [LDR3DGS (Baseline)](Link2)  

2. Run the following scripts in order
```shell
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

## Results 

![Alt text](https://github.com/shreyesss/HDRSplat-3DGS-for-HDR-scene-reconstruction/blob/main/assets/results.jpg)
![Alt text](https://github.com/shreyesss/HDRSplat-3DGS-for-HDR-scene-reconstruction/blob/main/assets/denoise.jpg)
![Alt text](https://github.com/shreyesss/HDRSplat-3DGS-for-HDR-scene-reconstruction/blob/main/assets/pointcloud.jpg)


## Acknowledgement 
The authors of this paper wish to express their gratitude to the following works for their significant contributions to the field, which have greatly enabled and inspired our research.
[RawNeRF](https://bmild.github.io/rawnerf/)
[3D gaussain splatting](https://github.com/graphdeco-inria/gaussian-splatting/tree/main)
[PMRID](https://github.com/MegEngine/PMRID)

## Concurrent Work 

There's a lot of excellent work that was introduced around the same time as ours.

[HDR-GS](https://arxiv.org/abs/2405.15125) also introduces HDR space 3D reconstructions.

[LE3D](https://arxiv.org/abs/2406.06216) uses a color-MLP explicitly unlike ours to represent RAW color space.






