import torch 
from raw_enhancement.PMRID.models.net_torch import Network 
import os 
import rawpy 
import random
import json
import numpy as np
from raw_enhancement.PMRID.utils import RawUtils



# print(raw.shape, raw.min(), raw.max())

# img_path = "/media/cilab/data/shreyas/PMRID/PMRID/PMRID/Scene1/dark/RAW_2020_02_27_17_49_05_875/input.raw"
# def read_array(path):
#     return np.fromfile(str(path), dtype=np.uint16)
# raw = read_array(img_path)
# print(raw.shape)
# print(raw.min(), raw.max())

def numpy_to_tensor(numpy_array):
    return torch.tensor(numpy_array, dtype = torch.float32)

def tensor_to_numpy(torch_tensor):
    return torch_tensor.numpy().astype(np.float32)

class KSigma:

    def __init__(self, K_coeff, B_coeff, anchor, V = 959.0):
        self.K = np.poly1d(K_coeff)
        self.Sigma = np.poly1d(B_coeff)
        self.anchor = anchor
        self.V = V

    def __call__(self, img_01, iso: float, inverse=False):
        k, sigma = self.K(iso), self.Sigma(iso)
        k_a, sigma_a = self.K(self.anchor), self.Sigma(self.anchor)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        img = img_01 * self.V

        if not inverse:
            img = img * cvt_k + cvt_b
        else:
            img = (img - cvt_b) / cvt_k

        return img / self.V


class Denoiser:

    def __init__(self, ksigma, inp_scale=256.0):
     
        net = Network()
        checkpoint = torch.load("/home/cilab/shreyas/codes/gaussian-splatting/raw_enhancement/PMRID/models/torch_pretrained.ckp")
        net.load_state_dict(checkpoint)
        net.eval()


        self.net = net
        self.ksigma = ksigma
        self.inp_scale = inp_scale

    def pre_process(self, bayer_01: np.ndarray):
        rggb = RawUtils.bayer2rggb(bayer_01)
        rggb = rggb.clip(0, 1)

        H, W = rggb.shape[:2]
        ph, pw = (32-(H % 32))//2, (32-(W % 32))//2
        rggb = np.pad(rggb, [(ph, ph), (pw, pw), (0, 0)], 'constant')
        inp_rggb = rggb.transpose(2, 0, 1)[np.newaxis]
        self.ph, self.pw = ph, pw
        return inp_rggb

    def run(self, bayer_01: np.ndarray, iso: float):
        inp_rggb_01 = self.pre_process(bayer_01)
        inp_rggb = self.ksigma(inp_rggb_01, iso) * self.inp_scale

        inp = np.ascontiguousarray(inp_rggb)
        # print(inp.shape)
        inp_tensor = numpy_to_tensor(inp)
        pred = self.net(inp_tensor)[0] / self.inp_scale

        # import ipdb; ipdb.set_trace()
        pred = pred.detach().numpy().transpose(1, 2, 0)
        pred = self.ksigma(pred, iso, inverse=True)

        ph, pw = self.ph, self.pw
        pred = pred[ph:-ph, pw:-pw]
        return RawUtils.rggb2bayer(pred)

def load_raw_exif(base_path):
    image_name = random.sample(os.listdir(base_path), 1)[0].split(".")[0]
    print(image_name)
    base = os.path.join(base_path,image_name)
    with open(base + '.dng', 'rb') as f:
      raw = rawpy.imread(f).raw_image
    with open(base + '.json', 'rb') as f:
      exif = json.load(f)[0]
    return raw, exif

    
def denoise(raw_image, iso):

    ksigma = KSigma(
        K_coeff=[0.0005995267, 0.00868861],
        B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
        anchor=1600,
    )
    denoiser = Denoiser(ksigma)
    # img_folder = "/media/cilab/data/shreyas/rawner-undistorted/scenes/gardenlights/raw"
    # raw, meta = load_raw_exif(img_folder)
    # # method 1 of rescaling
    # blacklevel = np.array(meta['BlackLevel']).reshape(-1,1,1)
    # whitelevel = np.array(meta['WhiteLevel']).reshape(-1,1,1)
    # raw_inp1 = ((raw - blacklevel) / (whitelevel - blacklevel) )[0]
    # print(raw_inp1.shape)
    denoised_raw_image = denoiser.run(raw_image, iso)
    return denoised_raw_image
    # method 2 of rescaling 
    
  

# run_benchmark()

# # import rawpy
# # import numpy as np

# def process_raw_image(raw_array_path, output_path):
#     # Read the raw image
#     raw_data = np.load(raw_array_path)

#     # Convert the raw array to a rawpy object
#     raw_image = rawpy.imread(raw_data)

#     # Perform post-processing (e.g., demosaicing, white balance, etc.)
#     rgb_image = raw_image.postprocess()

#     # Save the processed image
#     with open(output_path, 'wb') as f:
#         f.write(rgb_image)

# # Example usage:
# raw_array_path = "output1.npy"
# output_path = "output1.jpg"
# process_raw_image(raw_array_path, output_path)
# raw_array_path = "output2.npy"
# output_path = "output2.jpg"
# process_raw_image(raw_array_path, output_path)

