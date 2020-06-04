import os
import numpy as np
from math import floor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import transform
from skimage import io
from skimage.data import shepp_logan_phantom
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from skimage import img_as_ubyte

SCALING_FACTOR = 2

def main():
    # Get image list
    hrlist = []
    srlist = []
    for parent, dirnames, filenames in os.walk('data/test'):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                hrlist.append(os.path.join(parent, filename))
                srlist.append(os.path.join('raisr', 'results', 'filter_BSDS500', filename))

    num_images = len(hrlist)
    print('Processing {} images'.format(num_images))

    all_psnr = []
    all_ssim = []
    for hr_path, sr_path in zip(hrlist, srlist):
        hr = io.imread(hr_path)
        sr = io.imread(sr_path)
        hr_gray = img_as_ubyte(rgb2gray(hr))
        sr_gray = img_as_ubyte(rgb2gray(sr))

        psnr = peak_signal_noise_ratio(hr_gray, sr_gray)
        ssim = structural_similarity(hr_gray, sr_gray)
        print('{}: PSNR {}, SSIM {}'.format(os.path.basename(hr_path), psnr, ssim))

        all_psnr.append(psnr)
        all_ssim.append(ssim)
    
    print('Avg PSNR: {}'.format(np.mean(all_psnr)))
    print('Avg SSIM: {}'.format(np.mean(all_ssim)))


if __name__ == "__main__":
    main()
