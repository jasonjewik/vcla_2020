import argparse
from subprocess import call
import os
import glob
from natsort import natsorted
import pickle

from utils import progress_bar


def invoke_darknet(crop_path, darknet_dir):
    # Run Darknet
    # not tested on windows since I can't get darknet to run on Windows
    call(["bash",
          "darknet.sh",
          darknet_dir,
          "darknet_files/yolov3_10000.weights",
          crop_path,
          "darknet_files/obj.data",
          "darknet_files/yolov3.cfg"])
    print()


def label_brakes(crop_path):
    # Get the files
    filepaths = os.path.join(crop_path, "*.txt")
    files = natsorted(glob.glob(filepaths))

    brake_or_not = dict()
    brake = False

    # Check if darknet detected a brakelight
    for i, fi in enumerate(files):
        progress_bar(i, len(files))
        f = open(fi)
        for line in f:
            segments = line.split(' ')
            if segments[0] == "2":
                brake = True
                break

        brake_or_not[fi] = brake

        f.close()

    # write out to pickle file
    res_pkl = os.path.join(crop_path, "results.pkl")
    with open(res_pkl, 'wb') as res:
        pickle.dump(brake_or_not, res)


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='Runs darknet detector on the given images'
    )
    parser.add_argument('darknet_dir', action='store',
                        help='where the darknet executable is located')
    parser.add_argument('crop_path', action='store',
                        help='should be GTAV_program/drivedata/crop')
    args = parser.parse_args()

    if not os.path.isdir(args.crop_path):
        print('the specified data folder does not exist')
        print('please gather some data first')
        exit(1)

    invoke_darknet(args.crop_path, args.darknet_dir)
    label_brakes(args.crop_path)
