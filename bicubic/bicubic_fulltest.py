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

scaling_factor = 3
os.makedirs('lr/' + str(scaling_factor), exist_ok=True)
downscaled_imgs = []

imagelist = []
for parent, dirnames, filenames in os.walk('../data/IXI-T2/test'):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))

for img_path in imagelist:
    # print('Reading {}'.format(img_path))
    image = io.imread(img_path)
    image = rgb2gray(image)
    downscaled = downscale(image, scaling_factor)
    downscaled_imgs.append(img_as_ubyte(downscaled))

# record metrics
metricspath = 'results/{}/metrics.txt'.format(os.path.basename('bicubic'))
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
    image = img_as_ubyte(image)

    print('\r', end='')
    print(' ' * 60, end='')
    print('\rUpscaling image ' + str(imagecount) + ' of ' + str(len(imagelist)) + ' (' + img_path + ')')
    # origin = cv2.imread(image)
    origin = image
    starttime = time.time()

    result = cv2.resize(image, (hr_gray.shape[1], hr_gray.shape[0]), interpolation=cv2.INTER_CUBIC)

    endtime = time.time()
    timedelta = endtime - starttime
    print()
    metrics.write(str(timedelta) + '\n')

    # hr_gray = img_as_ubyte(hr_gray)
    # sr_gray = img_as_ubyte(rgb2gray(result))
    # sr_gray = rgb2gray(result)
    sr_gray = result/np.max(result)
    hr_gray = hr_gray/np.max(hr_gray)


    psnr = peak_signal_noise_ratio(hr_gray, sr_gray)
    ssim = structural_similarity(hr_gray, sr_gray)
    mprint('{}: PSNR {}, SSIM {}'.format(os.path.basename(img_path), psnr, ssim))

    all_psnr.append(psnr)
    all_ssim.append(ssim)
    all_timings.append(timedelta)
    
    # if args.write:
    #     cv2.imwrite('results/{}/'.format(os.path.basename(filtername)) + os.path.splitext(os.path.basename(img_path))[0] + '.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    # hr = cv2.imread('../data/test/{}'.format(os.path.basename(image)))
    # psnr = peak_signal_noise_ratio(hr, result)
    # ssim = structural_similarity(hr, result)
    # all_psnr.append(psnr)
    # all_ssim.append(ssim)
    # all_timings.append(timedelta)
    # print('PSNR: {}; SSIM: {}; Time: {}'.format(psnr, ssim, timedelta))
    # metrics.write('PSNR: {}; SSIM: {}; Time: {}\n'.format(psnr, ssim, timedelta))

    imagecount += 1
    # Visualizing the process of RAISR image upscaling
    # if args.plot:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(1, 4, 1)
    #     ax.imshow(grayorigin, cmap='gray', interpolation='none')
    #     ax = fig.add_subplot(1, 4, 2)
    #     ax.imshow(upscaledLR, cmap='gray', interpolation='none')
    #     ax = fig.add_subplot(1, 4, 3)
    #     ax.imshow(predictHR, cmap='gray', interpolation='none')
    #     ax = fig.add_subplot(1, 4, 4)
    #     ax.imshow(result, interpolation='none')
    #     plt.show()


mprint('')
mprint('Avg PSNR: {}'.format(np.mean(all_psnr)))
mprint('Avg SSIM: {}'.format(np.mean(all_ssim)))
mprint('Total time: {:.2f} sec, Avg time: {:.2f} sec'.format(np.sum(all_timings), np.mean(all_timings)))

print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')

metrics.close()
