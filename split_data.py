import os, sys, argparse
from random import shuffle
import numpy as np

def split_data(input_path, test, val):
    file_list = os.listdir(input_path)
    for files in file_list:
        if not files.lower().endswith(('.nii', '.nii.gz')):
            file_list.remove(files)
    shuffle(file_list)
    train_list, test_list, val_list = np.split(file_list, [int((1 - test - val) * len(file_list)), int((1 - val) * len(file_list))])
    	
    os.makedirs(os.path.join(input_path, "train"))
    os.makedirs(os.path.join(input_path, "test"))
    os.makedirs(os.path.join(input_path, "val"))

    for train_file in train_list:
        os.rename(os.path.join(input_path, train_file), os.path.join(input_path, "train", train_file))
    for test_file in test_list:
        os.rename(os.path.join(input_path, test_file), os.path.join(input_path, "test", test_file))
    for val_file in val_list:
        os.rename(os.path.join(input_path, val_file), os.path.join(input_path, "val", val_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="set input directory")
    parser.add_argument("-t", "--test", help="set test ratio")
    parser.add_argument("-v", "--validation", help="set validation ratio")

    args = parser.parse_args()

    input_path = "data\IXI-T2"
    test = 0.1
    val = 0.2

    if args.input:
        input_path = args.input

    if args.test:
        test = args.test

    if args.validation:
        val = args.validation
    
    split_data(input_path, test, val)