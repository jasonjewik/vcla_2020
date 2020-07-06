import cv2
import os
import ntpath as path
import pickle
import numpy as np
from natsort import natsorted
import argparse

from utils import progress_bar


def __read_pickle_files__(source_folder_path):
    imgs = list()
    filelist = natsorted(os.listdir(source_folder_path))
    num_files = len(filelist)

    print('reading pickle files')

    for i, file in enumerate(filelist):
        progress_bar(i, num_files)
        filepath = path.join(source_folder_path, file)

        if path.isfile(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                imgs.append(data)

    print('read all pickle files')

    return imgs


def generate_images(dst_folder, imgs):
    print('generating images')
    num_imgs = len(imgs)

    for i, img in enumerate(imgs):
        progress_bar(i, num_imgs)
        cv2.imwrite(path.join(
            dst_folder, f'{str(i).zfill(9)}_img.jpg'), img)
    print(f'successfully converted to images, stored in {dst_folder}')


def generate_video(dst_folder, imgs):
    print('generating video')
    num_imgs = len(imgs)

    height, width, layers = imgs[1].shape
    fps = 10

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidpath = path.join(dst_folder, 'video.avi')
    video = cv2.VideoWriter(vidpath, fourcc, fps, (width, height))

    for i in range(len(imgs)):
        progress_bar(i, num_imgs)
        video.write(imgs[i])

    cv2.destroyAllWindows()
    video.release()
    print(f'successfully created video, stored in {vidpath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate visuals (images or videos) from pickle files.'
    )
    parser.add_argument('source_folder', action='store',
                        help='should be GTAV_program\\drivedata\\pickles')
    parser.add_argument('-i', '--images', action='store_true',
                        help='whether to create images')
    parser.add_argument('-v', '--video', action='store_true',
                        help='whether to create video')
    args = parser.parse_args()

    if not path.isdir(args.source_folder):
        print('the specified source folder does not exist')
        print('please run `generate_pickle.py` first')
        exit(1)

    if not args.images and not args.video:
        print('please specify either -i, -v, or both')
        exit(1)

    dst_folder = path.join(args.source_folder, '..', 'imvid')

    # clear out dst_folder if already exists
    if not path.exists(dst_folder):
        os.mkdir(dst_folder)
    else:
        filelist = os.listdir(dst_folder)
        for file in filelist:
            filepath = path.join(dst_folder, file)
            name, ext = path.splitext(filepath)
            if args.images and ext == '.jpg':
                os.remove(filepath)
            if args.video and ext == '.avi':
                os.remove(filepath)

    imgs = __read_pickle_files__(args.source_folder)

    if args.images:
        generate_images(dst_folder, imgs)

    if args.video:
        generate_video(dst_folder, imgs)
