# Script for converting YOLO annotations previously used for darknet
# into a form that is usable for pytorch classification.
# NOT intended for end usage, but kept just in case.

import os.path as osp
import glob
from mmcv import list_from_file
import cv2
import pickle
import warnings

root_folder = ""
dst_folder = ""
data_folders = ("data01", "data02", "data03",
                "data04", "data05", "data06", "data07")
ON = 0
OFF = 1

idx = 0
for folder in data_folders:
    data_path = osp.join(root_folder, folder)
    img_paths = osp.join(data_path, "*_img.jpg")
    ann_paths = osp.join(data_path, "*_img.txt")

    img_files = glob.glob(img_paths)
    ann_files = glob.glob(ann_paths)

    for imf in img_files:
        name = osp.splitext(imf)[0]
        expected_ann_file = name + ".txt"
        assert expected_ann_file in ann_files, "Missing annotation file"

        res = {}
        im = cv2.imread(imf)
        res["image"] = im

        ann_list = list_from_file(expected_ann_file)
        if len(ann_list) != 2:
            warnings.warn(f"{expected_ann_file} is weird, skipping")
            continue

        for line in ann_list:
            arr = line.split(" ")

            if arr[0] == "0":
                res["label"] = ON
                break
            elif arr[0] == "1":
                res["label"] = OFF
                break

        dst_path = osp.join(dst_folder, str(idx).zfill(9) + ".pkl")
        with open(dst_path, "wb") as dst:
            pickle.dump(res, dst)
        idx += 1
