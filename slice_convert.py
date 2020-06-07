import os, sys, argparse
import pydicom
import nibabel
import cv2

def nii_to_png(input_path): #, output_path):
    # if not os.path.exists(output_path):
    #         os.makedirs(output_path)

    file_list = []
    for parent, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            if filename.lower().endswith(('.nii', '.nii.gz', '.dcm')):
               file_list.append(os.path.join(parent, filename))

    for files in file_list:
        if files.lower().endswith('.dcm'):
            image = pydicom.dcmread(files)
            img_name = files.replace('.dcm', '.png')
            cv2.imwrite(img_name, image.pixel_array)

        if files.lower().endswith(('.nii', '.nii.gz')):
            pixel_array = nibabel.load(files).get_fdata()
            total_slices = pixel_array.shape[2]
            for slices in range(0, total_slices):
                if slices % 2 == 0 and slices <= 40:
                    image =  pixel_array[:, :, slices]
                    img_name = files.replace('.nii.gz', '_{}.png'.format(slices))
                    if files.lower().endswith('.nii'):
                        img_name = files.replace('.nii', '_{}.png'.format(slices))
                    cv2.imwrite(img_name, image)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="set input directory")
    # parser.add_argument("-o", "--output", help="set output directory")

    args = parser.parse_args()

    input_path = "data\IXI-T2"
    # output_path = os.path.join(input_path, "converted")

    if args.input:
        input_path = args.input

    # if args.output:
    #     output_path = args.output

    nii_to_png(input_path) #, output_path)