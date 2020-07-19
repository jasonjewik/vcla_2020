import pickle
import argparse
import os
import cv2
import numpy as np
import glob
from natsort import natsorted

from utils import progress_bar


def display_pickles(result_path):
    # Get the files
    filepaths = os.path.join(result_path, "*.pkl")
    files = natsorted(glob.glob(filepaths))

    # Parse files
    for fi in files:
        f = open(fi, 'rb')
        pkl = pickle.load(f)
        images = pkl[:-1]
        data = pkl[-1]

        print(data)

        for im in images:
            cv2.imshow("window", im)
            cv2.waitKey()
            exit(1)


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='Checks the results'
    )
    parser.add_argument('result_path', action='store',
                        help='should be GTAV_program/results')
    args = parser.parse_args()

    if not os.path.isdir(args.result_path):
        print('the specified result folder does not exist')
        print('please generate the results first')
        exit(1)

    display_pickles(args.result_path)
