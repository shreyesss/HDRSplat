# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for processing and loading raw image data."""

import glob
import json
import os
import types
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from internal import image as lib_image
from internal import math
from internal import utils
import jax
import jax.numpy as jnp
import numpy as np
import rawpy
import torch 
import pickle 

import cv2

_Array = Union[np.ndarray, jnp.ndarray]
_Axis = Optional[Union[int, Tuple[int, ...]]]


def load_raw_exif(image_dir, image_name):
    base = os.path.join(image_dir, os.path.splitext(image_name)[0])
    #NOTE temoporary
  
    with open(base + '.dng', 'rb') as f:
      raw = rawpy.imread(f).raw_image
    # # else:
    # with open(base + '.CR2', 'rb') as f:
    #   raw = rawpy.imread(f).raw_image
    with open(base + '.json', 'rb') as f:
      exif = json.load(f)
    return raw, exif

def load_exif(image_dir, image_name):
    base = os.path.join(image_dir, os.path.splitext(image_name)[0])
    with open(base + '.json', 'rb') as f:
      exif = json.load(f)
    return exif



def postprocess_raw_cpu(raw: Union[np.ndarray, torch.Tensor],
                    camtorgb: Union[np.ndarray, torch.Tensor],
                    exposure: Optional[float] = None) -> Union[np.ndarray, torch.Tensor]:
    """Converts demosaicked raw to sRGB with a minimal postprocessing pipeline.

    Args:
        raw: [H, W, 3], demosaicked raw camera image.
        camtorgb: [3, 3], color correction transformation to apply to raw image.
        exposure: color value to be scaled to pure white after color correction.
                  If None, "autoexposes" at the 97th percentile.

    Returns:
        srgb: [H, W, 3], color corrected + exposed + gamma mapped image.
    """
    if raw.shape[-1] != 3:
        raise ValueError(f'raw.shape[-1] is {raw.shape[-1]}, expected 3')
    if camtorgb.shape != (3, 3):
        raise ValueError(f'camtorgb.shape is {camtorgb.shape}, expected (3, 3)')

    # Determine if input is NumPy or PyTorch tensor
    is_numpy = isinstance(raw, np.ndarray)
    is_torch = isinstance(raw, torch.Tensor)
    # print("torch" if is_torch else "not torch")
    # print("numpy" if is_numpy else "not numpy")
       

    # # Convert from camera color space to standard linear RGB color space.
    # transpose = torch.transpose if is_torch else np.transpose
    matmul = torch.matmul if is_torch else np.matmul

    camtorgb = torch.Tensor(camtorgb).t() if is_torch else np.array(camtorgb).T
    if is_torch:
      device = raw.device
      if camtorgb.device != device:
              camtorgb = camtorgb.to(device)
   
    rgb_linear = matmul(raw, camtorgb)
    if exposure is None:
        exposure = np.percentile(rgb_linear, 97) if is_numpy else torch.percentile(rgb_linear, 97)
    
    # "Expose" image by mapping the input exposure level to white and clipping.
    rgb_linear_scaled = np.clip(rgb_linear / exposure, 0, 1) if is_numpy else torch.clip(rgb_linear / exposure, 0, 1)
    
    # Apply sRGB gamma curve to serve as a simple tonemap.
    if is_numpy:
      srgb = lib_image.linear_to_srgb(rgb_linear_scaled)  
    else:
       srgb = lib_image.linear_to_srgb_torch(rgb_linear_scaled)

    return srgb

def postprocess_raw_gpu(raw: Union[np.ndarray, torch.Tensor],
                    camtorgb: Union[np.ndarray, torch.Tensor],
                    exposure: Optional[float] = None) -> Union[np.ndarray, torch.Tensor]:
    """Converts demosaicked raw to sRGB with a minimal postprocessing pipeline.

    Args:
        raw: [3, H, W], demosaicked raw camera image.
        camtorgb: [3, 3], color correction transformation to apply to raw image.
        exposure: color value to be scaled to pure white after color correction.
                  If None, "autoexposes" at the 97th percentile.

    Returns:
        srgb: [3, H, W], color corrected + exposed + gamma mapped image.
    """

   
    device = raw.device
    C,H,W = raw.shape
    raw = raw.reshape(C,H*W)
    camtorgb = torch.Tensor(camtorgb).to(device)
    rgb_linear = torch.matmul(camtorgb,raw)
    rgb_linear = rgb_linear.reshape(C,H,W)

     
    if exposure is None:
        exposure = torch.percentile(rgb_linear, 97)
    
    # "Expose" image by mapping the input exposure level to white and clipping.
    rgb_linear_scaled =  torch.clip(rgb_linear / exposure, 0, 1)

    
    # Apply sRGB gamma curve to serve as a simple tonemap.
   
    srgb = lib_image.linear_to_srgb_torch(rgb_linear_scaled)

    return srgb



def pixels_to_bayer_mask(pix_x: np.ndarray, pix_y: np.ndarray) -> np.ndarray:
  """Computes binary RGB Bayer mask values from integer pixel coordinates."""
  # Red is top left (0, 0).
  r = (pix_x % 2 == 0) * (pix_y % 2 == 0)
  # Green is top right (0, 1) and bottom left (1, 0).
  g = (pix_x % 2 == 1) * (pix_y % 2 == 0) + (pix_x % 2 == 0) * (pix_y % 2 == 1)
  # Blue is bottom right (1, 1).
  b = (pix_x % 2 == 1) * (pix_y % 2 == 1)
  return np.stack([r, g, b], -1).astype(np.float32)


def bilinear_demosaic(bayer: _Array,
                      xnp: types.ModuleType) -> _Array:
  """Converts Bayer data into a full RGB image using bilinear demosaicking.

  Input data should be ndarray of shape [height, width] with 2x2 mosaic pattern:
    -------------
    |red  |green|
    -------------
    |green|blue |
    -------------
  Red and blue channels are bilinearly upsampled 2x, missing green channel
  elements are the average of the neighboring 4 values in a cross pattern.

  Args:
    bayer: [H, W] array, Bayer mosaic pattern input image.
    xnp: either numpy or jax.numpy.

  Returns:
    rgb: [H, W, 3] array, full RGB image.
  """
  def reshape_quads(*planes):
    """Reshape pixels from four input images to make tiled 2x2 quads."""
    planes = xnp.stack(planes, -1)
    shape = planes.shape[:-1]
    # Create [2, 2] arrays out of 4 channels.
    zup = planes.reshape(shape + (2, 2,))
    # Transpose so that x-axis dimensions come before y-axis dimensions.
    zup = xnp.transpose(zup, (0, 2, 1, 3))
    # Reshape to 2D.
    zup = zup.reshape((shape[0] * 2, shape[1] * 2))
    return zup

  def bilinear_upsample(z):
    """2x bilinear image upsample."""
    # Using np.roll makes the right and bottom edges wrap around. The raw image
    # data has a few garbage columns/rows at the edges that must be discarded
    # anyway, so this does not matter in practice.
    # Horizontally interpolated values.
    zx = .5 * (z + xnp.roll(z, -1, axis=-1))
    # Vertically interpolated values.
    zy = .5 * (z + xnp.roll(z, -1, axis=-2))
    # Diagonally interpolated values.
    zxy = .5 * (zx + xnp.roll(zx, -1, axis=-2))
    return reshape_quads(z, zx, zy, zxy)

  def upsample_green(g1, g2):
    """Special 2x upsample from the two green channels."""
    z = xnp.zeros_like(g1)
    z = reshape_quads(z, g1, g2, z)
    alt = 0
    # Grab the 4 directly adjacent neighbors in a "cross" pattern.
    for i in range(4):
      axis = -1 - (i // 2)
      roll = -1 + 2 * (i % 2)
      alt = alt + .25 * xnp.roll(z, roll, axis=axis)
    # For observed pixels, alt = 0, and for unobserved pixels, alt = avg(cross),
    # so alt + z will have every pixel filled in.
    return alt + z

  r, g1, g2, b = [bayer[(i//2)::2, (i%2)::2] for i in range(4)]
  r = bilinear_upsample(r)
  # Flip in x and y before and after calling upsample, as bilinear_upsample
  # assumes that the samples are at the top-left corner of the 2x2 sample.
  b = bilinear_upsample(b[::-1, ::-1])[::-1, ::-1]
  g = upsample_green(g1, g2)
 
  rgb = xnp.stack([r, g, b], -1)

  # rgb = xnp.stack([r[::2], g[::2], b[::2]], -1)
  return rgb


bilinear_demosaic_jax = jax.jit(lambda bayer: bilinear_demosaic(bayer, xnp=jnp))
bilinear_demosaic_np = lambda bayer: bilinear_demosaic(bayer, xnp= np)


def load_raw_images(image_dir,
                    image_names,):
  """Loads raw images and their metadata from disk.

  Args:
    image_dir: directory containing raw image and EXIF data.
    image_names: files to load (ignores file extension), loads all DNGs if None.

  Returns:
    A tuple (images, exifs).
    images: [N, height, width, 3] array of raw sensor data.
    exifs: [N] list of dicts, one per image, containing the EXIF data.
  Raises:
    ValueError: The requested `image_dir` does not exist on disk.
  """

  if not utils.file_exists(image_dir):
    raise ValueError(f'Raw image folder {image_dir} does not exist.')

  # Load raw images (dng files) and exif metadata (json files).
  def load_raw_exif(image_name):
    base = os.path.join(image_dir, os.path.splitext(image_name)[0])
    with utils.open_file(base + '.dng', 'rb') as f:
      raw = rawpy.imread(f).raw_image
    with utils.open_file(base + '.json', 'rb') as f:
      exif = json.load(f)[0]
    return raw, exif

  if image_names is None:
    image_names = [
        os.path.basename(f)
        for f in sorted(glob.glob(os.path.join(image_dir, '*.dng')))
    ]

  data = [load_raw_exif(x) for x in image_names]
  raws, exifs = zip(*data)
  raws = np.stack(raws, axis=0).astype(np.float32)

  return raws, exifs


exposure_percentile = 97



# Brightness percentiles to use for re-exposing and tonemapping raw images.
_PERCENTILE_LIST = (80, 90, 97, 99, 100)

# Relevant fields to extract from raw image EXIF metadata.
# For details regarding EXIF parameters, see:
# https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/dng_spec_1.4.0.0.pdf.
_EXIF_KEYS = (
    'BlackLevel',  # Black level offset added to sensor measurements.
    'WhiteLevel',  # Maximum possible sensor measurement.
    'AsShotNeutral',  # RGB white balance coefficients.
    'ColorMatrix2',  # XYZ to camera color space conversion matrix.
    'NoiseProfile',  # Shot and read noise levels.
    'ISO'
)

# Color conversion from reference illuminant XYZ to RGB color space.
# See http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html.
_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])



def process_exif(exif):
  """Processes list of raw image EXIF data into useful metadata dict.

  Input should be a list of dictionaries loaded from JSON files.
  These JSON files are produced by running
    $ exiftool -json IMAGE.dng > IMAGE.json
  for each input raw file.

  We extract only the parameters relevant to
  1. Rescaling the raw data to [0, 1],
  2. White balance and color correction, and
  3. Noise level estimation.

  Args:
    exifs: a list of dicts containing EXIF data as loaded from JSON files.

  Returns:
    meta: a dict of the relevant metadata for running RawNeRF.
  """
  meta = {}
  print(exif['ShutterSpeed'])
  #exif = exif[0]
  # Convert from array of dicts (exifs) to dict of arrays (meta).
  for key in _EXIF_KEYS:
    exif_value = exif.get(key)
    if exif_value is None:
      continue
    # Values can be a single int or float...
    if isinstance(exif_value, int) or isinstance(exif_value, float):
      vals = exif[key]
    # Or a string of numbers with ' ' between.
    elif isinstance(exif_value, str):
      vals = [float(z) for z in exif[key].split(' ')]
    meta[key] = np.squeeze(np.array(vals))
  # Shutter speed is a special case, a string written like 1/N.
  # NOTE Fix array shape later
  if exif['ShutterSpeed'] == 8:
      meta['ShutterSpeed'] = np.array(8)
  elif exif['ShutterSpeed'] == 1/2:
      meta['ShutterSpeed'] = np.array(0.5)
  else:
      meta['ShutterSpeed'] = np.array(0.0125)
#  meta['ShutterSpeed'] = np.array(1. / float(exif['ShutterSpeed'].split('/')[1]))
  # print("AAAAAAAAAAAAAAAAAAAAAAAAA")
  # print(exif['ShutterSpeed'], meta['ShutterSpeed'])
  # Create raw-to-sRGB color transform matrices. Pipeline is:
  # cam space -> white balanced cam space ("camwb") -> XYZ space -> RGB space.
  # 'AsShotNeutral' is an RGB triplet representing how pure white would measure
  # on the sensor, so dividing by these numbers corrects the white balance.
  whitebalance = meta['AsShotNeutral'].reshape(-1, 3)
  cam2camwb = np.array([np.diag(1. / x) for x in whitebalance])
  # ColorMatrix2 converts from XYZ color space to "reference illuminant" (white
  # balanced) camera space.
  xyz2camwb = meta['ColorMatrix2'].reshape(-1, 3, 3)
  rgb2camwb = xyz2camwb @ _RGB2XYZ
  # We normalize the rows of the full color correction matrix, as is done in
  # https://github.com/AbdoKamel/simple-camera-pipeline.
  rgb2camwb /= rgb2camwb.sum(axis=-1, keepdims=True)
  # Combining color correction with white balance gives the entire transform.
  cam2rgb = np.linalg.inv(rgb2camwb) @ cam2camwb
  meta['cam2rgb'] = cam2rgb

  return meta


def generate_meta(exif,raw):
    meta = process_exif(exif)
    shutter_ratio = 1.
   
   
    blacklevel = meta['BlackLevel'].reshape(-1, 1, 1)
    whitelevel = meta['WhiteLevel'].reshape(-1, 1, 1)
    # print("meta",meta["BlackLevel"],meta["WhiteLevel"])
    # print("raw",raw.min(), raw.max())
    raw_image = (raw - blacklevel) / (whitelevel - blacklevel) * shutter_ratio
    # print("raw_img",raw_image.min(),raw_image.max())
    def demosaic_fn(x):
          x_jax = jnp.array(x)
          x_demosaic_jax = bilinear_demosaic_jax(x_jax)
          return np.array(x_demosaic_jax)

    raw_image_demosaiced = demosaic_fn(raw_image[0])
    # raw_image_demosaiced = bilinear_demosaic_np(raw_image[0])
    raw_rgb = raw_image_demosaiced @ meta['cam2rgb'][0].T
    exposure = np.percentile(raw_rgb, exposure_percentile)
    meta['exposure'] = exposure
    # Sweep over various exposure percentiles to visualize in training logs.
    exposure_levels = {p: np.percentile(raw_rgb, p) for p in _PERCENTILE_LIST}
    meta['exposure_levels'] = exposure_levels

    # Create postprocessing function mapping raw images to tonemapped sRGB space.
    cam2rgb0 = meta['cam2rgb'][0]
    postprocess_fn = lambda z, x=exposure: postprocess_raw_cpu(z, cam2rgb0, x)
    return meta, raw_image, raw_image_demosaiced, postprocess_fn




def load_raw_dataset(denoise_method, scene_path, raw_folder, image_name):

  preprocssed_path = os.path.join(scene_path,"denoised")
  # print(denoise_method)

  if os.path.exists(preprocssed_path):
    if denoise_method == "PMRID":
        # print("in pmrid")
        raw_image_path = os.path.join(preprocssed_path, "PMRID_raw", image_name.split(".")[0]+".npy")
        raw_image_denoised = np.load(raw_image_path).astype(np.float32)
        meta_path = os.path.join(preprocssed_path, "metadata", image_name.split(".")[0]+".pkl")
        with open(meta_path, 'rb') as f:
          meta = pickle.load(f)
        postprocess_fn = lambda z, x=meta['exposure']: postprocess_raw_cpu(z, meta['cam2rgb'][0], x)

        return raw_image_denoised, meta, postprocess_fn

    if denoise_method == "bilateral":
        raw_image_path = os.path.join(preprocssed_path, "bilateral_raw2", image_name.split(".")[0]+".npy")
        raw_image_denoised = np.load(raw_image_path).astype(np.float32)
        meta_path = os.path.join(preprocssed_path, "metadata", image_name.split(".")[0]+".pkl")
        with open(meta_path, 'rb') as f:
          meta = pickle.load(f)
        postprocess_fn = lambda z, x=meta['exposure']: postprocess_raw_cpu(z, meta['cam2rgb'][0], x)

        return raw_image_denoised, meta, postprocess_fn

    if denoise_method == "bm3d":
        raw_image_path = os.path.join(preprocssed_path, "bm3d_raw", image_name.split(".")[0]+".npy")
        raw_image_denoised = np.load(raw_image_path).astype(np.float32)
        meta_path = os.path.join(preprocssed_path, "metadata", image_name.split(".")[0]+".pkl")
        with open(meta_path, 'rb') as f:
          meta = pickle.load(f)
        postprocess_fn = lambda z, x=meta['exposure']: postprocess_raw_cpu(z, meta['cam2rgb'][0], x)

        return raw_image_denoised, meta, postprocess_fn
    
  

    if denoise_method == "median":
        raw_image_path = os.path.join(preprocssed_path, "median_raw", image_name.split(".")[0]+".npy")
        meta_path = os.path.join(preprocssed_path, "metadata", image_name.split(".")[0]+".pkl")
        raw_image_denoised = np.load(raw_image_path).astype(np.float32)
        with open(meta_path, 'rb') as f:
          meta = pickle.load(f)
        postprocess_fn = lambda z, x=meta['exposure']: postprocess_raw_cpu(z, meta['cam2rgb'][0], x)

        return raw_image_denoised, meta, postprocess_fn

    if denoise_method == "demosaic":
        raw_image_path = os.path.join(preprocssed_path, "demosaiced_raw", image_name.split(".")[0]+".npy")
        meta_path = os.path.join(preprocssed_path, "metadata", image_name.split(".")[0]+".pkl")
        raw_image_denoised = np.load(raw_image_path).astype(np.float32)
        with open(meta_path, 'rb') as f:
          meta = pickle.load(f)
        postprocess_fn = lambda z, x=meta['exposure']: postprocess_raw_cpu(z, meta['cam2rgb'][0], x)

        return raw_image_denoised, meta, postprocess_fn
    else:
      raw, exif = load_raw_exif(raw_folder, image_name)
      meta, raw_image, raw_image_demosaiced, postprocess_fn = generate_meta(exif,raw)

      
      # def demosaic_fn(x):
      #     x_jax = jnp.array(x)
      #     x_demosaic_jax = bilinear_demosaic_jax(x_jax)
      
      #     return np.array(x_demosaic_jax)
      
      # raw_image = demosaic_fn(raw_image[0])
      return raw_image_demosaiced, meta, postprocess_fn

def load_raw_dataset_from_scratch(scene_path, raw_folder, image_name):
      raw, exif = load_raw_exif(raw_folder, image_name)
      meta,raw_image,raw_image_demosaiced, postprocess_fn = generate_meta(exif,raw)
      return raw_image, raw_image_demosaiced, meta, postprocess_fn
   
   



def best_fit_affine(x: _Array, y: _Array, axis: _Axis) -> _Array:
  """Computes best fit a, b such that a * x + b = y, in a least square sense."""
  x_m = x.mean(axis=axis)
  y_m = y.mean(axis=axis)
  xy_m = (x * y).mean(axis=axis)
  xx_m = (x * x).mean(axis=axis)
  # slope a = Cov(x, y) / Cov(x, x).
  a = (xy_m - x_m * y_m) / (xx_m - x_m * x_m)
  b = y_m - a * x_m
  return a, b


def match_images_affine(est: _Array, gt: _Array,
                        axis: _Axis = (0, 1)) -> _Array:
  """Computes affine best fit of gt->est, then maps est back to match gt."""
  # Mapping is computed gt->est to be robust since `est` may be very noisy.
  a, b = best_fit_affine(gt, est, axis=axis)
  # Inverse mapping back to gt ensures we use a consistent space for metrics.
  est_matched = (est - b) / a
  return est_matched
