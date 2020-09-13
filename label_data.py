import argparse
import os
import glob
import shutil
import random
from subprocess import call
from utils.read_raw_data import read_raw_data
from utils.generate_visuals import generate_images, generate_images_and_video
from utils.crop_images import crop_images
from utils.parse_turns import parse_turns
from utils.detect_brakes import detect_brakes

# Determine operating system
if os.name == "nt":
    OS = "windows"
elif os.name == "posix":
    OS = "linux"
else:
    OS = "unknown"

print(f'Detected OS: {OS}')

# Parse arguments
parser = argparse.ArgumentParser(
    description='Labels image sequences of the motorcycle driving in GTA V'
)
parser.add_argument('root_folder', action='store',
                    help='should be GTAV_program/drivedata')
parser.add_argument('checkpoint_file', action='store',
                    help='should be wherever the pytorch checkpoint file is; see README')
parser.add_argument('-p', '--parent', action='store_true',
                    help='whether the root folder is a parent folder, see README \
                    for details')
parser.add_argument('-r', '--resize', action='store_true', default=False,
                    help='if specified, the result pickles will contain 1280x720 images \
                    scaled down to 320x180')
parser.add_argument('-c', '--crop', action='store_true', default=False,
                    help='if specified, the result pickles will contain 400x400 images \
                    cropped from the original 1280x720 images')
parser.add_argument('-o', '--open', action='store_true',
                    help='whether to open directory when done')
parser.add_argument('-s', '--system', action='store', metavar="OS", default=OS,
                    help='what OS you are on (this program should be able to \
                    determine, but just in case); please specify "windows" \
                    or "linux"')
parser.add_argument('-v', '--video', action='store_true',
                    help='whether to create video')
parser.add_argument('-k', '--keep', action='store_true', default=False,
                    help='if specified, the program will not clean up any \
                    intermediate files')
parser.add_argument('-n', '--num_samples_per_npz', action='store', default=500,
                    help='number of samples put into each npz archive in the output')
args = parser.parse_args()

# Check folder input
root_folder = args.root_folder
if not os.path.isdir(root_folder):
    print('the specified root folder does not exist')
    exit(1)

# Check for output type
if args.resize is False and args.crop is False:
    print('please specify either --resize, --crop, or both')
    exit(1)

# Get the child directories
directories = []
if args.parent:
    print('found the following child directories: ')
    files = glob.glob(os.path.join(root_folder, '*'))
    for f in files:
        if os.path.isdir(f):
            directories.append(f)

    # ask user for confirmation
    while True:
        for d in directories:
            print(f'> {os.path.split(d)[-1]}')
        user_input = input(
            'Are these the child directories you want to use? ([Y]/n) ') or True

        if user_input is True or user_input == 'Y':
            break
        elif user_input == 'n':
            print('Stepping through the child directories one at a time...')
            for i, d in enumerate(directories):
                while True:
                    user_input = input(
                        f'Do you want to use "{os.path.split(d)[-1]}"? ([Y]/n) ') or True
                    if user_input is True or user_input == 'Y':
                        break
                    elif user_input == 'n':
                        directories[i] = None
                        break
                    else:
                        print('invalid input')
            directories = [d for d in directories if d is not None]
        else:
            print('invalid input')

# if no child folders were added, add root_folder to the directories list
if len(directories) == 0:
    directories = [root_folder]

# Check OS
if args.system == "windows":
    OS = "windows"
elif args.system == "linux":
    OS = "linux"
else:
    print(f'unrecognized OS provided: {args.system}')
    exit(1)

# Loop through directories
num_steps = 5
for dirnum, direc in enumerate(directories):
    curr_step = 1
    print(f'=== DIRECTORY {dirnum + 1} OF {len(directories)} ===')

    # Read data
    print(f'=> STEP {curr_step}/{num_steps}')
    read_raw_data(direc)
    curr_step += 1

    # Generate images
    print(f'=> STEP {curr_step}/{num_steps}')
    pickle_dir = os.path.join(direc, 'pickles')
    if not args.video:
        generate_images(pickle_dir)
    else:
        generate_images_and_video(pickle_dir)
    image_dir = os.path.join(direc, 'imvid')
    curr_step += 1

    # crop images
    print(f'=> STEP {curr_step}/{num_steps}')
    crop_images(image_dir)
    curr_step += 1

    # run brake detection
    print(f'=> STEP {curr_step}/{num_steps}')
    crop_dir = os.path.join(direc, 'crop')
    detect_brakes(crop_dir, args.checkpoint_file)
    curr_step += 1

    # Parse turns
    print(f'=> STEP {curr_step}/{num_steps}')
    parse_turns(direc, args.resize, args.crop, args.num_samples_per_npz)
    curr_step += 1

    print('all data has been labeled')

    # Clean up intermediate files
    if args.keep is False:
        print('cleaning up intermediate files')
        shutil.rmtree(pickle_dir)
        shutil.rmtree(image_dir)
        shutil.rmtree(crop_dir)
        print('done with cleanup')

    # Open workspace
    if args.open and not args.parent:
        print("opening workspace")
        if OS == "linux":
            call(['xdg-open', direc])
        elif OS == "windows":
            call(['explorer', direc])
        else:
            print(f"I'm not sure how, but we made a mistake here")

# Open parent directory if open arg was specified
if args.open and args.parent:
    print("opening workspace")
    if OS == "linux":
        call(['xdg-open', root_folder])
    elif OS == "windows":
        call(['explorer', root_folder])
    else:
        print(f"I'm not sure how, but we made a mistake here")
