import glob
import os
import csv
import math
import numpy as np
import cv2
from natsort import natsorted
import argparse
import vg
import pickle
import random

if __name__ == '__main__':  # assumes the program is run from utils directory
    from progress_bar import progress
else:  # assumes the program is run from label_data.py
    from utils.progress_bar import progress


def __convert_paths__(impath):
    direc, image = os.path.split(impath)
    parent_direc, _ = os.path.split(direc)
    new_parent_direc = os.path.join(parent_direc, 'imvid')
    return os.path.join(new_parent_direc, image)


def __parse_turns__(drivedata, imagedata, workspace_path, brakedata_array, output_dir, num_samples_per_npz):
    # Check if the number of images matches the number of drivedata CSVs
    len_drive = len(drivedata)
    len_image = len(imagedata)

    if len_image > len_drive:
        imagedata = imagedata[:len_drive]
        print('more image data than drive data')
    elif len_drive > len_image:
        drivedata = drivedata[:len_image]
        print('more drive data than image data')
    else:
        print('amount of drive data and image data matches')

    # Create result directory if not exist, otherwise clear it
    result_dir = os.path.join(workspace_path, output_dir)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        print(f'created directory {result_dir}')
    else:
        filelist = os.listdir(result_dir)
        for file in filelist:
            filepath = os.path.join(result_dir, file)
            os.remove(filepath)
        print(f'cleared directory {result_dir}')

    # Create pickle files
    print('generating results')

    num_items = len(imagedata)
    output = []
    count = 0

    # Get the image size
    first_img = cv2.imread(imagedata[0])
    image_shape = np.asarray(first_img.shape)

    for i in range(num_items):
        progress(i, num_items)

        positions = np.empty((5, 3))
        headings = np.empty((5, 3))
        images = []
        braking = 0

        if (i + 1 > num_items - 5):
            break

        # Get image data in sequence
        for j in range(i, i + 5):

            # Append images
            imf = cv2.imread(imagedata[j])
            if output_dir == 'results_resized':
                imf = cv2.resize(
                    imf, (image_shape[1] // 4, image_shape[0] // 4))
            elif output_dir == 'results_cropped':
                imf = cv2.resize(imf, (image_shape[1], image_shape[0]))
            images.append(imf)

            # Get brake data values
            if (brakedata_array[j] is True):
                braking += 1

            # Open drivedata CSV file
            df = open(drivedata[j], 'r')
            reader = csv.reader(df, delimiter=' ')

            # Get position data
            row = next(reader)
            x = float(row[0][1:])
            y = float(row[1])
            z = float(row[2][:-1])
            positions[j - i] = np.array([x, y, z])

            # Get heading data
            row = next(reader)
            u = float(row[0][1:])
            v = float(row[1])
            w = float(row[2][:-1])
            headings[j - i] = np.array([u, v, w])

            df.close()

        # changes in position
        del_pos = vg.euclidean_distance(positions[1:], positions[:-1])

        # changes in heading
        del_head = vg.signed_angle(
            headings[:-1], headings[1:], look=vg.basis.z, units='rad')

        # find average change in heading
        avg_head_change = float(np.average(del_head))

        # find average change in position
        avg_pos_change = float(np.average(del_pos))

        # determine keys
        W = 0
        A = 0
        S = 0
        D = 0

        # detect acceleration
        if (braking >= 3 and avg_pos_change >= 0.1):
            S = 1
        else:
            for p1, p2 in zip(del_pos[:-1], del_pos[1:]):
                if p2 > p1:
                    W = 1
                elif p2 < p1:
                    W = 0
                    break

        # detect turns
        if (avg_head_change <= -0.03):
            D = 1
        elif (avg_head_change >= 0.03):
            A = 1

        # save to npz file
        images.append([A, D, W, S])
        output.append(images)
        if len(output) % num_samples_per_npz == 0:
            outpath = os.path.join(result_dir, f'data{count}.npz')
            np.savez_compressed(outpath, np.asarray(output, dtype=object))
            count += 1
            output = []

    progress(num_items, num_items)
    print(f'done writing results to {result_dir}')


def parse_turns(workspace_path, resize, crop, num_samples_per_npz):
    # Get the drive and image data
    drivedata_path = os.path.join(workspace_path, "*.csv")
    drivedata = natsorted(glob.glob(drivedata_path))

    # Get the parsed brake data
    brakedata_path = os.path.join(
        workspace_path, "crop", "brake_detect_results.pkl")
    brakedata_file = open(brakedata_path, 'rb')
    brakedata_contents = pickle.load(brakedata_file)
    brakedata_array = list(brakedata_contents.values())

    # Determine the images to use
    if resize:
        imagedata = list(map(__convert_paths__, brakedata_contents.keys()))
        __parse_turns__(drivedata, imagedata, workspace_path,
                        brakedata_array, 'results_resized', num_samples_per_npz)
    if crop:
        imagedata = list(brakedata_contents.keys())
        __parse_turns__(drivedata, imagedata, workspace_path,
                        brakedata_array, 'results_cropped', num_samples_per_npz)


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='Parses turns of the given data'
    )
    parser.add_argument('workspace_path', action='store',
                        help='should be GTAV_program/drivedata')
    parser.add_argument('-r', '--resize', action='store_true', default=False,
                        help='if specified, the result pickles will contain 1280x720 images \
                    scaled down to 320x180')
    parser.add_argument('-c', '--crop', action='store_true', default=False,
                        help='if specified, the result pickles will contain 400x400 images \
                        cropped from the original 1280x720 images')
    parser.add_argument('-n', '--num_samples_per_npz', action='store', default=500, type=int,
                        help='number of samples put into each npz archive in the output')
    args = parser.parse_args()

    if not os.path.isdir(args.workspace_path):
        print('the specified data folder does not exist')
        print('please gather some data first')
        exit(1)

    if args.resize is False and args.crop is False:
        print('please specify either --resize, --crop, or both')
        exit(1)

    parse_turns(args.workspace_path, args.resize,
                args.crop, args.num_samples_per_npz)
