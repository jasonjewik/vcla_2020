"""
NOT CURRENTLY WORKING
"""

import pickle
import argparse
import os
import cv2
import numpy as np
import glob
from natsort import natsorted

# assumes the program is run from scripts directory
import sys
sys.path.insert(0, os.path.realpath('..'))
if __name__ == '__main__':
    from utils.progress_bar import progress


def display_pickles(result_path):
    # Get the files
    filepaths = os.path.join(result_path, "*.pkl")
    files = natsorted(glob.glob(filepaths))

    # Parse files
    for ind, fi in enumerate(files):
        progress(ind, len(files), fraction=True)
        f = open(fi, 'rb')
        pkl = pickle.load(f)
        images = pkl[:-1]
        data = pkl[-1]
        text = f'A: {data[0]}, D: {data[1]}, W: {data[2]}, S: {data[3]}'

        # text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2

        # truncated file name
        fname = os.path.split(fi)[-1]

        for im in images:
            image = cv2.putText(im, text, org, font, fontScale,
                                color, thickness, cv2.LINE_AA, False)
            cv2.imshow(fname, image)
            key = cv2.waitKey(0)

            if key == ord('q'):
                print()
                f.close()
                exit()

        cv2.destroyWindow(fname)
        f.close()


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='Checks the results'
    )
    parser.add_argument('result_path', action='store',
                        help='should be GTAV_program/results_resized or \
                        GTAV_program/results_cropped')
    args = parser.parse_args()

    if not os.path.isdir(args.result_path):
        print('the specified result folder does not exist')
        print('please generate the results first')
        exit(1)

    print("press 'q' to quit")
    display_pickles(args.result_path)
