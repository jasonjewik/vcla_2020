import argparse
import os
import torch
from natsort import natsorted
import glob
import cv2
import pickle
import time

if __name__ == '__main__':  # assumes the program is run from utils directory
    from progress_bar import progress
    import sys
    sys.path.insert(0, os.path.realpath('..'))
else:  # assumes the program is run from label_data.py
    from utils.progress_bar import progress

from pytorch_classifier.net import Net
from pytorch_classifier.prepare_data import prepare_data


def detect_brakes(crop_folder, checkpoint_file):
    # Load the model
    net = Net()
    net.load_state_dict(torch.load(checkpoint_file))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # Get images
    cropped_path = os.path.join(crop_folder, "*.jpg")
    files = natsorted(glob.glob(cropped_path))

    # Results dict
    results = {}

    # Pass images into model
    print("getting predictions from network")
    start_time = time.time()
    for i, f in enumerate(files):
        progress(i, len(files))

        raw_img = cv2.imread(f)
        inputs = prepare_data(raw_img).to(device)
        outputs = net(inputs)

        predicted = torch.max(outputs.data, 1)[1].item()

        if predicted == 0:
            results[f] = True
        elif predicted == 1:
            results[f] = False

    end_time = time.time()

    avg_time = (end_time - start_time)/len(files) * 1000
    print(f'avg time per image: {avg_time:.3f}ms')

    # Write results to pickle file
    dest_file = 'brake_detect_results.pkl'
    dest_path = os.path.join(crop_folder, dest_file)
    with open(dest_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'wrote results to {dest_path}')


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='Detects brakes in the cropped images, writing to brake_detect_results.pkl'
    )
    parser.add_argument('crop_folder', action='store',
                        help='should be GTAV_program/drivedata/crop')
    parser.add_argument('checkpoint_file', action='store',
                        help='should be checkpoints/__.pth')
    args = parser.parse_args()

    if not os.path.isdir(args.crop_folder):
        print('the specified source folder does not exist')
        print('please run crop_images.py first')
        exit(1)

    if not os.path.isfile(args.checkpoint_file):
        print('the checkpoint file does not exist')
        exit(1)

    detect_brakes(args.crop_folder, args.checkpoint_file)
