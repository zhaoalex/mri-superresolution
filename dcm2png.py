import os
import pydicom
import cv2

def dcm_to_png(data_path='data/HMRI Yannan/'):
    img_path = os.listdir(data_path)
    for image in img_path:
        if '.dcm' in image:
            dcm = pydicom.dcmread(os.path.join(data_path, image))
            image = image.replace('.dcm', '.png')
            target_path = os.path.join(data_path, 'converted/')
            os.makedirs(os.path.join(target_path))
            cv2.imwrite(os.path.join(target_path, image), dcm.pixel_array)

if __name__ == "__main__":
    data_path = int(sys.argv[1])
    dcm_to_png(data_path)
    main()