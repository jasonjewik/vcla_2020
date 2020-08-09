import cv2
import os
import pickle
import numpy as np
from natsort import natsorted
import argparse

if __name__ == '__main__':  # assumes the program is run from utils directory
    from progress_bar import progress
else:  # assumes the program is run from label_data.py
    from utils.progress_bar import progress


def __read_pickle_files__(source_folder_path):
    imgs = list()
    filelist = natsorted(os.listdir(source_folder_path))
    num_files = len(filelist)

    print('reading pickle files')

    for i, file in enumerate(filelist):
        progress(i, num_files)
        filepath = os.path.join(source_folder_path, file)

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                imgs.append(data)

    print('read all pickle files')

    return imgs


def __generate_images__(dst_folder, imgs):
    print('generating images')
    num_imgs = len(imgs)

    for i, img in enumerate(imgs):
        progress(i, num_imgs)
        cv2.imwrite(os.path.join(
            dst_folder, f'{str(i).zfill(9)}_img.jpg'), img)
    print(f'successfully converted to images, stored in {dst_folder}')


def __generate_video__(dst_folder, imgs):
    print('generating video')
    num_imgs = len(imgs)

    height, width = imgs[1].shape[:2]
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidpath = os.path.join(dst_folder, 'video.avi')
    video = cv2.VideoWriter(vidpath, fourcc, fps, (width, height))

    for i in range(len(imgs)):
        progress(i, num_imgs)
        video.write(imgs[i])

    cv2.destroyAllWindows()
    video.release()
    print(f'successfully created video, stored in {vidpath}')


def generate_images(src_folder):
    imgs = __read_pickle_files__(src_folder)

    dst_folder = os.path.join(src_folder, '..', 'imvid')
    dst_folder = os.path.abspath(dst_folder)

    # clear out dst_folder if already exists
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
    else:
        filelist = os.listdir(dst_folder)
        for file in filelist:
            filepath = os.path.join(dst_folder, file)
            os.remove(filepath)

    __generate_images__(dst_folder, imgs)


def generate_video(src_folder):
    dst_folder = os.path.join(src_folder, '..', 'imvid')
    dst_folder = os.path.abspath(dst_folder)
    imgs = __read_pickle_files__(src_folder)
    __generate_video__(dst_folder, imgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate visuals (images or videos) from pickle files.'
    )
    parser.add_argument('source_folder', action='store',
                        help='should be GTAV_program/drivedata/pickles')
    parser.add_argument('-i', '--images', action='store_true',
                        help='whether to create images')
    parser.add_argument('-v', '--video', action='store_true',
                        help='whether to create video')
    args = parser.parse_args()

    if not os.path.isdir(args.source_folder):
        print('the specified source folder does not exist')
        print('please run `generate_pickle.py` first')
        exit(1)

    if not args.images and not args.video:
        print('please specify either -i, -v, or both')
        exit(1)

    dst_folder = os.path.join(args.source_folder, '..', 'imvid')
    dst_folder = os.path.abspath(dst_folder)

    # clear out dst_folder if already exists
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
    else:
        filelist = os.listdir(dst_folder)
        for file in filelist:
            filepath = os.path.join(dst_folder, file)
            name, ext = os.path.splitext(filepath)
            if args.images and ext == '.jpg':
                os.remove(filepath)
            if args.video and ext == '.avi':
                os.remove(filepath)

    imgs = __read_pickle_files__(args.source_folder)

    if args.images:
        __generate_images__(dst_folder, imgs)

    if args.video:
        __generate_video__(dst_folder, imgs)
