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

if __name__ == '__main__':  # assumes the program is run from utils directory
    from progress_bar import progress
else:  # assumes the program is run from label_data.py
    from utils.progress_bar import progress


def __convert_paths__(impath):
    direc, image = os.path.split(impath)
    parent_direc, _ = os.path.split(direc)
    new_parent_direc = os.path.join(parent_direc, 'imvid')
    return os.path.join(new_parent_direc, image)


def parse_turns(workspace_path):
    # Get the drive and image data
    drivedata_path = os.path.join(workspace_path, "*.csv")
    drivedata = natsorted(glob.glob(drivedata_path))

    # Get the parsed brake data
    brakedata_path = os.path.join(
        workspace_path, "crop", "brake_detect_results.pkl")
    brakedata_file = open(brakedata_path, 'rb')
    brakedata_contents = pickle.load(brakedata_file)
    brakedata_array = list(brakedata_contents.values())

    # Get paths to the uncropped images
    imagedata = list(map(__convert_paths__, brakedata_contents.keys()))

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

    for i in range(0, num_items):
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

        # save to pickle file
        output = copy.copy(images)
        output.append([A, D, W, S])
        outfile_path = os.path.join(result_dir, f'{str(i).zfill(9)}_res.pkl')

        out = open(outfile_path, 'wb')
        pickle.dump(output, out)

    progress(num_items, num_items)
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
