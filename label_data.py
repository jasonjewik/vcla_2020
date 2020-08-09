import argparse
import os
import glob
import shutil
from generate_pickles import generate_pickles
from generate_visuals import generate_images, generate_video
from crop_images import crop_images
from parse_turns import parse_turns
from invoke_darknet import invoke_darknet, label_brakes
from subprocess import call

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
    description='Label training data'
)
parser.add_argument('root_folder', action='store',
                    help='should be GTAV_program/drivedata')
parser.add_argument('darknet_dir', action='store',
                    help='should be wherever the darknet executable file is')
parser.add_argument('-v', '--video', action='store_true',
                    help='whether to create video')
parser.add_argument('-p', '--parent', action='store_true',
                    help='whether the root folder is a parent folder, \
                    see README for details')
parser.add_argument('-o', '--open', action='store_true',
                    help='whether to open directory when done')
parser.add_argument('-s', '--system', action='store', metavar="OS", default=OS,
                    help='what OS you are on (this program should be able to \
                    determine, but just in case); please specify "windows" \
                    or "linux"')
parser.add_argument('-k', '--keep', action='store_true', default=False,
                    help='if specified, the program will not clean up any \
                    intermediate files')
args = parser.parse_args()

# Check folder input
root_folder = args.root_folder
if not os.path.isdir(root_folder):
    print('the specified root folder does not exist')
    exit(1)

# Get the child directories
directories = []
if args.parent:
    print('found the following child directories: ')
    files = glob.glob(os.path.join(root_folder, '*'))
    for f in files:
        if os.path.isdir(f):
            directories.append(f)
            print(f)

    # ask user for confirmation
    confirm = True
    while True:
        user_input = input(
            'Are these the folders you want to use? ([Y]/n) ') or confirm
        if user_input == confirm or user_input == 'Y':
            break
        elif user_input == 'n':
            print('exiting')
            exit(1)
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
for direc in directories:
    # Generate images
    generate_pickles(direc)
    pickle_dir = os.path.join(direc, 'pickles')
    generate_images(pickle_dir)
    image_dir = os.path.join(direc, 'imvid')

    # Optional: create video
    if args.video:
        generate_video(pickle_dir)

    # crop images
    crop_images(image_dir)
    crop_dir = os.path.join(direc, 'crop')

    # run darknet detection
    invoke_darknet(crop_dir, args.darknet_dir)
    label_brakes(crop_dir)

    # Parse turns
    parse_turns(direc)

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
        call(['xdg-open', direc])
    elif OS == "windows":
        call(['explorer', direc])
    else:
        print(f"I'm not sure how, but we made a mistake here")
