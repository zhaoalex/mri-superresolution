from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import ToTensor
import time
import os
from math import floor
from skimage import transform
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import numpy as np

# ===========================================================
# Argument settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('-s', '--upscale_factor', type=str, default='2', help='Set scaling factor')
parser.add_argument('-i', '--input', type=str, required=False, default='../data/test', help='input directory to use')
parser.add_argument('-m', '--model', type=str, default='FSRCNN', help='model to use')
parser.add_argument('-w', '--write', action='store_true')
args = parser.parse_args()
print(args)

def mprint(out):
    print(out)
    metrics.write(str(out) + '\n')

def downscale(img, scaling_factor):
    if len(img.size) == 2:
        height, width = img.size
        size = (floor(height / scaling_factor), floor(width / scaling_factor))
    else:
        height, width, third = img.size
        size = (floor(height / scaling_factor), floor(width / scaling_factor), third)
    return img.resize(size, Image.BICUBIC)

# ===========================================================
# input image setting
# ===========================================================
GPU_IN_USE = torch.cuda.is_available()
device = torch.device('cuda' if GPU_IN_USE else 'cpu')
model = torch.load(os.path.join('models', args.model, args.upscale_factor, 'best_model.pth'), map_location=lambda storage, loc: storage)
model = model.to(device)

if GPU_IN_USE:
    cudnn.benchmark = True

imagelist = []
for parent, dirnames, filenames in os.walk(args.input):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))

# record metrics
metricspath = 'results/{}/metrics.txt'.format(args.model)
os.makedirs(os.path.dirname(metricspath), exist_ok=True)
metrics = open(metricspath, 'w+')

all_timings = []
all_psnr = []
all_ssim = []

for img_path in imagelist:
    print('Upscaling image {}'.format(img_path))

    hr = Image.open(img_path)
    hr_gray = hr.convert('L')
    img = downscale(hr_gray, int(args.upscale_factor))
    img = img.convert('YCbCr')

    y, cb, cr = img.split()
    data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    data = data.to(device)

    starttime = time.time()

    out = model(data)
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    endtime = time.time()
    timedelta = endtime - starttime
    metrics.write(str(timedelta) + '\n')

    # hr_gray = img_as_ubyte(hr_gray)
    # sr_gray = img_as_ubyte(rgb2gray(result))
    sr_gray = out_img.convert('L')

    hr_gray = np.array(hr_gray)
    sr_gray = np.array(sr_gray)

    psnr = peak_signal_noise_ratio(hr_gray, sr_gray)
    ssim = structural_similarity(hr_gray, sr_gray)
    mprint('{}: PSNR {}, SSIM {}'.format(os.path.basename(img_path), psnr, ssim))

    all_psnr.append(psnr)
    all_ssim.append(ssim)
    all_timings.append(timedelta)
    
    if args.write:
        out_path = 'results/{}/'.format(args.model) + os.path.splitext(os.path.basename(img_path))[0] + '.png'
        out_img.save(out_path)
        print('output image saved to ', out_path)

mprint('')
mprint('Avg PSNR: {}'.format(np.mean(all_psnr)))
mprint('Avg SSIM: {}'.format(np.mean(all_ssim)))
mprint('Total time: {:.2f} sec, Avg time: {:.2f} sec'.format(np.sum(all_timings), np.mean(all_timings)))

metrics.write(str(all_psnr) + '\n')
metrics.write(str(all_ssim) + '\n')
metrics.write(str(all_timings) + '\n')
