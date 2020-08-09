# Script to prepare images for passing into the pytorch classifier

import cv2
import torch
import numpy as np
import os
import pickle
import argparse
from natsort import natsorted
from subprocess import call
import sys
import shutil

if __name__ == '__main__':
    sys.path.insert(0, os.path.realpath(".."))
    from utils.read_raw_data import read_raw_data
    from utils.generate_visuals import generate_images
    from utils.crop_images import crop_images
    from utils.progress_bar import progress


def prepare_for_eval(img):
    """
    Assumes images as produced by utils/crop_images and read by cv2.imread.
    Based on TestPipeline in motorcycle_dataset.
    """

    # resize
    out_shape = (128, 128)
    img = cv2.resize(img, out_shape)

    # normalize
    zero_img = np.zeros(out_shape)
    norm_img = cv2.normalize(img, zero_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = norm_img / 255.0

    # convert to tensor
    tens = np.transpose(norm_img, (2, 1, 0))
    tens = torch.from_numpy(tens).float()

    # add dimension
    result = torch.unsqueeze(tens, 0)

    return result


def __prepare_for_training__(img, label, data_direc):
    """
    Creates pickle files to put into the data subdirectory.
    Pickle file format: dict{"image": image_data (np.ndarray), "label": on/off (0, 1)}
    Returns the file path of the image.
    """
    num_files = len(os.listdir(data_direc))

    assert type(img) is np.ndarray
    assert type(label) is int

    result = {"image": img, "label": label}
    res_path = os.path.join(data_direc, f'{str(num_files).zfill(9)}.pkl')

    with open(res_path, 'wb') as f:
        pickle.dump(result, f)

    return res_path


if __name__ == "__main__":
    # Determine operating system
    if os.name == "nt":
        OS = "windows"
    elif os.name == "posix":
        OS = "linux"
    else:
        OS = "unknown"

    print(f'Detected OS: {OS}')

    # Get arguments
    parser = argparse.ArgumentParser(
        description='Helps with labeling training data quickly'
    )
    parser.add_argument('root_folder', action='store',
                        help='should be GTAV_program/drivedata')
    parser.add_argument('-o', '--open', action='store_true',
                        help='whether to open the result directory when done')
    parser.add_argument('-s', '--system', action='store', metavar="OS", default=OS,
                        help='what OS you are on (this program should be able to \
                        determine, but just in case); please specify "windows" \
                        or "linux"')
    args = parser.parse_args()

    crop_folder = os.path.join(args.root_folder, "crop")
    crop_folder_exists = True if os.path.isdir(crop_folder) else False

    if not crop_folder_exists:
        print('the specified crop folder does not exist')
        print('running utilities to generate crop folder')
        pkl_direc = os.path.join(args.root_folder, "pickles")
        imgs_direc = os.path.join(args.root_folder, "imvid")

        read_raw_data(args.root_folder)
        generate_images(pkl_direc)
        crop_images(imgs_direc)

        print('removing original pickles and imvid folders')
        shutil.rmtree(pkl_direc)
        shutil.rmtree(imgs_direc)
        print()

    # Loop through the images
    print('== Instructions ==')
    print('1. press "A" to label the image as "brake light on"')
    print('2. press "D" to label the image as "brake light off"')
    print('3. press "Q" to quit')
    print('4. press "B" to go back one image')
    print('5. press any other key to skip')

    file_names = natsorted(os.listdir(crop_folder))
    files = []
    for name in file_names:
        files.append(os.path.join(crop_folder, name))

    data_direc = os.path.realpath('../data')
    output_paths = []
    i = 0

    while i in range(len(files)):
        progress(i, len(files), fraction=True)

        img = files[i]
        assert os.path.isfile(img)
        data = cv2.imread(img)
        cv2.imshow('', data)

        key = cv2.waitKey(0)

        if key == ord('a'):
            res_path = __prepare_for_training__(data, 0, data_direc)
            output_paths.append(res_path)
        elif key == ord('d'):
            res_path = __prepare_for_training__(data, 1, data_direc)
            output_paths.append(res_path)
        elif key == ord('q'):
            print()
            break
        elif key == ord('b'):
            if i > 0:
                i -= 1
            if len(output_paths) > 0:
                os.remove(output_paths.pop())
            continue

        i += 1

    print('done labeling data')
    print(f'first result saved to {output_paths[0]}')
    print(f'last result saved to {output_paths[-1]}')

    # open workspace
    if args.open:
        print("opening workspace")
        if OS == "linux":
            call(['xdg-open', data_direc])
        elif OS == "windows":
            call(['explorer', data_direc])
        else:
            print(f"I'm not sure how, but we made a mistake here")

    if not crop_folder_exists:
        # remove the crop folder if this program generated it
        # otherwise, leave it
        print('removing crop folder')
        shutil.rmtree(crop_folder)
