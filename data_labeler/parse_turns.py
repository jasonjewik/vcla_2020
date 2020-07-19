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

from utils import progress_bar


def parse_turns(workspace_path):
    # Get the drive and image data
    drivedata_path = os.path.join(workspace_path, "*.csv")
    imagedata_path = os.path.join(workspace_path, "imvid/*.jpg")
    drivedata = natsorted(glob.glob(drivedata_path))
    imagedata = natsorted(glob.glob(imagedata_path))

    # Check if the number of images matches the number of drivedata CSVs
    if len(imagedata) > len(drivedata):
        imagedata = imagedata[:len(drivedata)]
        print('more image data than drive data')
    elif len(drivedata) > len(imagedata):
        drivedata = drivedata[:len(imagedata)]
        print('more drive data than image data')
    else:
        print('number of drive data and image data matches')

    # Create results directory if not exist, otherwise clear it
    result_dir = os.path.join(workspace_path, "results")
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
    print('generating pickled results')

    num_items = len(drivedata)
    for i in range(0, num_items, 5):
        progress_bar(i, num_items)

        positions = np.empty([5, 3])
        headings = np.empty([5, 3])
        images = []

        if (i + 1 > num_items - 5):
            break

        # Get image data in sequence
        for j in range(i, i + 5):
            # Append images
            imf = cv2.imread(imagedata[j])
            images.append(imf)

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
        del_pos = vg.euclidean_distance(positions[:-1], positions[1:])

        # find average change in position
        avg_pos_change = float(np.average(del_pos))

        # changes in heading
        del_head = vg.signed_angle(
            headings[:-1], headings[1:], look=vg.basis.z, units='rad')

        # find average change in heading
        avg_head_change = float(np.average(del_head))

        # save to pickle file
        output = copy.copy(images)
        output.append([avg_pos_change, avg_head_change])
        outfile_path = os.path.join(result_dir, f'{str(i).zfill(9)}_res.pkl')

        out = open(outfile_path, 'wb')
        pickle.dump(output, out)

    print(f'done writing results to {result_dir}')


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='Parses turns of the given data'
    )
    parser.add_argument('workspace_path', action='store',
                        help='should be GTAV_program/drivedata')
    args = parser.parse_args()

    if not os.path.isdir(args.workspace_path):
        print('the specified data folder does not exist')
        print('please gather some data first')
        exit(1)

    parse_turns(args.workspace_path)
