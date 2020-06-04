import os
import sys
import cv2
import numpy as np
from math import floor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import transform
from skimage import io
from skimage.data import shepp_logan_phantom
from skimage import img_as_ubyte
from skimage.color import rgb2gray

def downscale(img, scaling_factor):
    if len(img.shape) == 2:
        height, width = img.shape
        size = (floor(height / scaling_factor), floor(width / scaling_factor))
    else:
        height, width, third = img.shape
        size = (floor(height / scaling_factor), floor(width / scaling_factor), third)
    return transform.resize(img, size, order=3, anti_aliasing=False)

def main(scaling_factor):
    # Get image list
    imagelist = []
    for parent, dirnames, filenames in os.walk('data/test'):
        for filename in filenames:
                if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    imagelist.append(os.path.join(parent, filename))

    for img_path in imagelist:
        print('Reading {}'.format(img_path))
        image = io.imread(img_path)
        image = rgb2gray(image)
        downscaled = downscale(image, scaling_factor)
        io.imsave('lr/' + str(scaling_factor) + '/' + os.path.splitext(os.path.basename(img_path))[0] + '.png', img_as_ubyte(downscaled))
        # cv2.imwrite('lr/' + str(scaling_factor) + '/' + os.path.splitext(os.path.basename(img_path))[0] + '.png', cv2.cvtColor(downscaled, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    scaling_factor = int(sys.argv[1])
    os.makedirs('lr/' + str(scaling_factor), exist_ok=True)
    main(scaling_factor)
