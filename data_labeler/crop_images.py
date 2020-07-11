import cv2
import os
import argparse
from natsort import natsorted

from utils import progress_bar

def crop_images(image_folder):
    if not os.path.isdir(image_folder):
        print('the specified image folder does not exist')
        print('please run `generate_visuals.py` first')
        exit(1)

    dst_folder = os.path.join(image_folder, '..', 'crop')
    dst_folder = os.path.abspath(dst_folder)

    # clear out dst_folder if already exists
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
    else:
        filelist = os.listdir(dst_folder)
        for file in filelist:
            filepath = os.path.join(dst_folder, file)
            filepath = os.path.join(image_folder, file)
            name, ext = os.path.splitext(filepath)
            if ext == '.jpg':
                os.remove(filepath)

    filelist = natsorted(os.listdir(image_folder))
    num_files = len(filelist)

    for i, file in enumerate(filelist):
        progress_bar(i, num_files)
        filepath = os.path.join(image_folder, file)
        name, ext = os.path.splitext(filepath)
        if ext == '.jpg':
            img = cv2.imread(filepath)
            crop_img = img[320:720, 440:840]
            cv2.imwrite(
                os.path.join(dst_folder, f'{str(i).zfill(9)}_img.jpg'), crop_img)

    print(f'successfully cropped images, stored in {dst_folder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Crops images generated by `generate_visuals.py`'
    )
    parser.add_argument('image_folder', action='store',
                        help='should be GTAV_program/drivedata/imvid')
    args = parser.parse_args()

    crop_images(args.image_folder)