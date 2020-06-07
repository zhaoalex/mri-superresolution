import cv2
import numpy as np
import os
import pickle
import sys
import time
from math import floor
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import transform
from skimage import io
from skimage import img_as_ubyte
from skimage.color import rgb2gray
import argparse

parser = argparse.ArgumentParser(description='Bicubic')
parser.add_argument('-s', '--scaling_factor', type=str, default='2', help='Set scaling factor')
parser.add_argument('-i', '--input', type=str, required=False, default='../data/test', help='input directory to use')
parser.add_argument('-w', '--write', action='store_true')
args = parser.parse_args()
print(args)

scaling_factor = int(args.scaling_factor)

def mprint(out):
    print(out)
    metrics.write(str(out) + '\n')

def downscale(img, scaling_factor):
    if len(img.shape) == 2:
        height, width = img.shape
        size = (floor(height / scaling_factor), floor(width / scaling_factor))
    else:
        height, width, third = img.shape
        size = (floor(height / scaling_factor), floor(width / scaling_factor), third)
    return transform.resize(img, size, order=3, anti_aliasing=False)


imagelist = []
for parent, dirnames, filenames in os.walk(args.input):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))

# record metrics
metricspath = 'results/{}/metrics.txt'.format(scaling_factor)
os.makedirs(os.path.dirname(metricspath), exist_ok=True)
metrics = open(metricspath, 'w+')

all_timings = []
all_psnr = []
all_ssim = []

imagecount = 1
for img_path in imagelist:
    hr = io.imread(img_path)
    hr_gray = rgb2gray(hr)
    image = downscale(hr_gray, scaling_factor)
    # image = img_as_ubyte(image)

    # make sure downscale will divide evenly
    old_scale = hr_gray.shape
    new_scale = ((old_scale[0] // scaling_factor) * scaling_factor, (old_scale[1] // scaling_factor) * scaling_factor)
    hr_gray = transform.resize(hr_gray, new_scale)

    print('Upscaling image ' + str(imagecount) + ' of ' + str(len(imagelist)) + ' (' + img_path + ')')
    starttime = time.time()

    result = transform.resize(image, hr_gray.shape, order=3)

    endtime = time.time()
    timedelta = endtime - starttime
    metrics.write(str(timedelta) + '\n')

    sr_gray = result
    hr_gray = hr_gray

    # # normalization?
    # sr_gray = result/np.max(result)
    # hr_gray = hr_gray/np.max(hr_gray)

    psnr = peak_signal_noise_ratio(hr_gray, sr_gray)
    ssim = structural_similarity(hr_gray, sr_gray)
    mprint('{}: PSNR {}, SSIM {}'.format(os.path.basename(img_path), psnr, ssim))

    all_psnr.append(psnr)
    all_ssim.append(ssim)
    all_timings.append(timedelta)

    if args.write:
        io.imsave('results/{}/'.format(scaling_factor) + os.path.splitext(os.path.basename(img_path))[0] + '.png', img_as_ubyte(result))
        # cv2.imwrite('results/{}/'.format(scaling_factor) + os.path.splitext(os.path.basename(img_path))[0] + '.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    imagecount += 1

mprint('')
mprint('Avg PSNR: {}'.format(np.mean(all_psnr)))
mprint('Avg SSIM: {}'.format(np.mean(all_ssim)))
mprint('Total time: {:.2f} sec, Avg time: {:.2f} sec'.format(np.sum(all_timings), np.mean(all_timings)))

print('Finished.')

metrics.close()
