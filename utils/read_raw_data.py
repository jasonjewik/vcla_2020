import os
import numpy as np
import cv2
import pickle
import argparse

if __name__ == '__main__':  # assumes the program is run from utils directory
    from progress_bar import progress
else:  # assumes the program is run from label_data.py
    from utils.progress_bar import progress

# expects images to be 1280x720
# thus, the resolution of GTA V should be 1280x720
# when recording data


def read_raw_data(root_folder_path):
    dst_folder = os.path.join(root_folder_path, 'pickles')

    if not os.path.isdir(dst_folder):
        os.mkdir(dst_folder)
    else:
        filelist = os.listdir(dst_folder)
        for file in filelist:
            filepath = os.path.join(dst_folder, file)
            os.remove(filepath)

    first_file_path = os.path.join(root_folder_path, 'RGB_whole.raw')
    if not os.path.exists(first_file_path):
        print(f'RGB_whole.raw does not exist in {root_folder_path}')
        print('please double check the directory')
        exit(1)

    print('reading RGB_whole.raw')
    rgbData = np.fromfile(first_file_path, dtype=np.uint8)
    rgbData = np.reshape(
        rgbData, (int(rgbData.shape[0]/1280/720/4), 720, 1280, 4))

    print('generating pickles')
    for i in range(rgbData.shape[0]):
        progress(i, rgbData.shape[0])
        rgb_img = rgbData[i]
        rgb_img = rgb_img[:, :, :3]
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        with open(os.path.join(dst_folder, f'{str(i).zfill(9)}_rgb.pkl'), 'wb') as fid:
            pickle.dump(rgb_img, fid, pickle.HIGHEST_PROTOCOL)

    print(f'successfully wrote pkl files to {dst_folder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extact pickle files from drivedata'
    )
    parser.add_argument('root_folder', action='store',
                        help='should be GTAV_program/drivedata')
    args = parser.parse_args()

    if not os.path.isdir(args.root_folder):
        print('the specified root folder does not exist')
        exit(1)

    read_raw_data(args.root_folder)
