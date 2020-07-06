import os
import ntpath as path
import numpy as np
import cv2
import pickle
import argparse

from utils import progress_bar

# expects images to be 1280x720
# thus, the resolution of GTA V should be 1280x720
# when recording data


def generate_pickles(root_folder_path, dst_folder):
    first_file_path = path.join(root_folder_path, 'RGB_whole.raw')

    rgbData = np.fromfile(first_file_path, dtype=np.uint8)
    rgbData = np.reshape(
        rgbData, (int(rgbData.shape[0]/1280/720/4), 720, 1280, 4))

    for i in range(rgbData.shape[0]):
        progress_bar(i, rgbData.shape[0])
        rgb_img = rgbData[i]
        rgb_img = rgb_img[:, :, :3]
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        with open(path.join(dst_folder, f'{str(i).zfill(9)}_rgb.pkl'), 'wb') as fid:
            pickle.dump(rgb_img, fid, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extact pickle files from drivedata'
    )
    parser.add_argument('root_folder', action='store',
                        help='should be GTAV_program\\drivedata')
    args = parser.parse_args()

    if not path.isdir(args.root_folder):
        print('the specified root folder does not exist')
        exit(1)

    dst_folder = path.join(args.root_folder, 'pickles')

    if not path.isdir(dst_folder):
        os.mkdir(dst_folder)
    else:
        filelist = os.listdir(dst_folder)
        for file in filelist:
            filepath = path.join(dst_folder, file)
            os.remove(filepath)

    generate_pickles(args.root_folder, dst_folder)
    print(f'successfully wrote pkl files to {dst_folder}')
