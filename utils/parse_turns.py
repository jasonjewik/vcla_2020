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
import copy
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


def parse_turns(workspace_path, resize, crop):
    # Get the drive and image data
    drivedata_path = os.path.join(workspace_path, "*.csv")
    drivedata = natsorted(glob.glob(drivedata_path))

    # Get the parsed brake data
    brakedata_path = os.path.join(
        workspace_path, "crop", "brake_detect_results.pkl")
    brakedata_file = open(brakedata_path, 'rb')
    brakedata_contents = pickle.load(brakedata_file)
    brakedata_array = list(brakedata_contents.values())

    # Determine whether to use the uncropped images or the cropped ones
    im_split = 0
    imagedata = list()
    if resize:
        imagedata.extend(map(__convert_paths__, brakedata_contents.keys()))
        im_split = len(imagedata)
    if crop:
        imagedata.extend(brakedata_contents.keys())

    # Check if the number of images matches the number of drivedata CSVs
    len_drive = len(drivedata)
    len_image = len(imagedata)
    if resize and crop:
        len_image //= 2

    if len_image > len_drive:
        imagedata = imagedata[:len_drive]
        print('more image data than drive data')
    elif len_drive > len_image:
        drivedata = drivedata[:len_image]
        print('more drive data than image data')
    else:
        print('number of drive data and image data matches')

    # Create results directories if not exist, otherwise clear it
    result_resized_dir = os.path.join(workspace_path, 'results_resized')
    result_cropped_dir = os.path.join(workspace_path, 'results_cropped')

    if resize:
        if not os.path.exists(result_resized_dir):
            os.mkdir(result_resized_dir)
            print(f'created directory {result_resized_dir}')
        else:
            filelist = os.listdir(result_resized_dir)
            for file in filelist:
                filepath = os.path.join(result_resized_dir, file)
                os.remove(filepath)
            print(f'cleared directory {result_resized_dir}')
    if crop:
        if not os.path.exists(result_cropped_dir):
            os.mkdir(result_cropped_dir)
            print(f'created directory {result_cropped_dir}')
        else:
            filelist = os.listdir(result_cropped_dir)
            for file in filelist:
                filepath = os.path.join(result_cropped_dir, file)
                os.remove(filepath)
            print(f'cleared directory {result_cropped_dir}')

    # Create pickle files
    print('generating pickled results')

    num_items = len(imagedata)

    for i in range(num_items):
        progress(i, num_items)

        positions = np.empty([5, 3])
        headings = np.empty([5, 3])
        images = []
        braking = 0

        if (i + 1 > num_items - 5):
            break

        # Get image data in sequence
        for j in range(i, i + 5):

            # Append images
            imf = cv2.imread(imagedata[j])
            
            # Resize images if the resize option is specified
            if resize and i < im_split:
                # 50% chance of putting a black rectangle in the corner (hiding the mini-map)
                # Assumes a 1280x720 image      
                if random.randint(0, 1) == 0:
                    imf = cv2.rectangle(imf, (0, 560), (210, 720), (0, 0, 0), -1)
                # Resize the image
                shape = imf.shape
                imf = cv2.resize(imf, (shape[1]//4, shape[0]//4))
            # We don't need to do anything if the crop option is specified

            images.append(imf)

            # Get brake data values
            idx = j
            if resize and crop and idx >= im_split:
                idx -= len_drive
            if (brakedata_array[idx] is True):
                braking += 1

            # Open drivedata CSV file
            df = open(drivedata[idx], 'r')
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

        # save to pickle file
        output = copy.copy(images)
        output.append([A, D, W, S])
        if resize and i < im_split:
            outfile_path = os.path.join(result_resized_dir, f'{str(i).zfill(9)}_res.pkl')
        if (crop and not resize) or (crop and resize and i >= im_split):
            outfile_path = os.path.join(result_cropped_dir, f'{str(i).zfill(9)}_res.pkl')

        out = open(outfile_path, 'wb')
        pickle.dump(output, out)

    progress(num_items, num_items)
    if resize:
        print(f'done writing results to {result_resized_dir}')
    if crop:
        print(f'done writing results to {result_cropped_dir}')


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
    args = parser.parse_args()

    if not os.path.isdir(args.workspace_path):
        print('the specified data folder does not exist')
        print('please gather some data first')
        exit(1)

    if args.resize is False and args.crop is False:
        print('please specify either --resize, --crop, or both')
        exit(1)

    parse_turns(args.workspace_path, args.resize, args.crop)
