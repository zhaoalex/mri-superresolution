import tarfile
from os import remove
from os.path import exists, join, basename

from six.moves import urllib
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, RandomHorizontalFlip
import torchvision.transforms.functional as TF
import random

from .dataset import DatasetFromFolder

class RandomDiscreteRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

def download_bsd300(dest="./dataset"):
    # output_image_dir = join(dest, "BSDS300/images")
    output_image_dir = join(dest, "../../data")

    # if not exists(output_image_dir):
    #     url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
    #     print("downloading url ", url)

    #     data = urllib.request.urlopen(url)

    #     file_path = join(dest, basename(url))
    #     with open(file_path, 'wb') as f:
    #         f.write(data.read())

    #     print("Extracting data")
    #     with tarfile.open(file_path) as tar:
    #         for item in tar:
    #             tar.extract(item, dest)

    #     remove(file_path)

    return output_image_dir


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        RandomDiscreteRotation([0, 90, 180, 270]),
        RandomHorizontalFlip(),
        Resize(crop_size // upscale_factor, interpolation=3),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        RandomDiscreteRotation([0, 90, 180, 270]),
        RandomHorizontalFlip(),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    root_dir = download_bsd300()
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    root_dir = download_bsd300()
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
