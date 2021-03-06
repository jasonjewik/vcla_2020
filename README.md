# VCLA Autonomous Vehicles in _GTA V_

## Quick Start

Please install the provided conda environment with the following command. If you are on a Linux machine, switch the environment file to `linux_env.yml`.

```
$ conda env create -f win_env.yml
```

Next, download and extract this [tar.gz archive](https://drive.google.com/file/d/1oGUJKXzhc7VVT5_20n9bs61CPyOyE7UO/view?usp=sharing) - the files contained within should go into the checkpoints folder.
Your directory structure should look like this:

```
~> /path/to/this/repository
   |-- checkpoints
   |   |-- motorcycle_net_epoch01.pth
   |   |-- ...other checkpoints...
   |   |-- motorcycle_net_epoch10.pth
   |-- data
   |   |-- ...pickled files...
   |-- pytorch_classifier
   |   |-- ...classifier files...
   |-- scripts
   |   |-- ...miscellaneous scripts...
   |-- utils
   |   |-- ...utility functions...
   |-- label_data.py
   |-- win_env.yml
   |-- linux_env.yml
   |-- README.md
```

To label data generated by the GTA V trainer, do the following commands:

```
> conda activate data_labeler
> python label_data.py /path/to/GTAV_program/drivedata checkpoints/motorcycle_net_epoch10.pth --open --resize
```

Results will appear in /path/to/GTAV_program/drivedata/results_resized/. These are intended to be used with https://github.com/jasonjewik/Self-Driving-Car-in-Video-Games/.
Please see the **Contents** section for advanced usage and how to train the Pytorch classifier.

## Contents

This section does not descibe _all_ contents of this repository. It only mentions files/directories of interest.

### `label_data.py`

- creates labeled images put into npz archives
- if the --resize flag is set, the output will go into GTAV_program/drivedata/results_resized
- if the --crop flag is set, the output will go into GTAV_program/drivedata/results_cropped
- if both flags are set, both results folders will be generated
- if the --parent flag is set, the script expects the following directory structure:
  ```
  ~> /path/to/root_folder/
     |-- child_folder_1
     |   |-- *.csv
     |   |-- RGB_whole.raw
     |-- ...other subdirectories...
     |-- child_folder_n
     |   |-- *.csv
     |   |-- RGB_whole.raw
  ```
- should be the only script you need to run
- if something goes wrong, try running the scripts in utils in this order:
  1. `read_raw_data.py`
  2. `generate_visuals.py`
  3. `crop_images.py`
  4. `detect_brakes.py`
  5. `parse_turns.py`

### utils directory

- `read_raw_data.py`

  - creates .pkl files from GTAV_program/drivedata/RGB_whole.raw
  - puts them into GTAV_program/drivedata/pickles
  - can take a long time before the progress bar is shown if the RGB_whole.raw file is very large

- `generate_visuals.py`

  - creates .jpg and/or .avi files from GTAV_program/drivedata/pickles/\*.pkl
  - puts them into GTAV_program/drivedata/imvid

- `crop_images.py`

  - crops the .jpg files in GTAV_program/drivedata/imvid/\*.jpg
  - puts them into GTAV_program/drivedata/crop
  - assumes a motorcycle as viewed in 3rd person mode

- `detect_brakes.py`

  - runs pytorch classifier to detect braking in cropped images
  - accepts files from GTAV_program/drivedata/crop/\*.jpg

- `parse_turns.py`

  - parse the turns based on data in GTAV_program/drivedata/\*.csv and the .jpg files in GTAV_program/drivedata/crop
  - generates npz archives and puts them in GTAV_program/drivedata/results
  - each archive consists of training samples
  - each training sample is five frames formatted as numpy arrays, followed by an array indicating the key presses
  - key presses are formatted like \[A, D, W, S\]

### scripts directory

- `graph_coords.py`

  - displays a graphical representation of the coordinates in GTAV_program/drivedata/\*.csv

- `test_results.py`

  - NOT CURRENTLY WORKING
  - checks the results in GTAV_program/drivedata/results
  - uses cv2 to show the images and predicted keys
  - press 'q' to quit at any time

### pytorch_classifier directory

- `README.md`
  - sample `prepare_data.py` output
  - training/testing logs for the pytorch classifier
- `prepare_data.py`
  - a tool for quickly labeling data
  - results are put into the top-level data subdirectory
- `train.py`
  - trains the classifier
- `test.py`
  - shows accuracy, precision, and recall scores of the classifier on the testing set
- `net.py`
  - definition for the classifier model

## To Do

- [x] Parse drivedata CSV files
- [x] Test on Windows
- [x] Train YOLOv3 model to determine braking/not-braking
- [x] Figure out why cv2.imshow has a GTK error on my machine
- [x] Run YOLOv3 model on cropped images to detect braking
- [x] Change `parse_turns.py` to print out keys
- [x] Change detection model to classification model
- [x] Change to pytorch model
- [x] Reorganize repostiroy contents
- [ ] Refactor code for consistency
- [x] Test on Linux
- [ ] Test classifier on data labeled with `prepare_data.py`

## Further Notes

- Previously, this repository used [Darknet](https://github.com/pjreddie/darknet) to do object detection for determining whether the motorcycle was braked
  - This version of the repository can still be found on the `darknet` branch
  - However, I could not get Darknet working on Windows, where the GTAV program is being run, even when using [AlexeyAB's fork](https://github.com/AlexeyAB/darknet)
- After Darknet could not be used, I tried switching to [mmdetection](https://github.com/open-mmlab/mmdetection), using [Dr. Wang's fork for Windows](https://github.com/kezewang/mmdetection)
  - However, I also ran into issues using mmdetection
- Eventually, I switched to using an image classifier built with Pytorch for a few reasons
  1. The problem is just a binary one: "brake light on" vs. "brake light off"
  2. We never expect to detect more than one brake light in a given image
  3. I'm fairly confident that the Pytorch model should run on both Linux and Windows without issue
  4. Reasonably high accuracy is achieved for very little training time (compared to the training time needed for Darknet)
- The only issue with switching to a Pytorch image classifier was reformatting the YOLO annotations
  - This is why training data for the model says
    - 0 = "brake light on"
    - 1 = "brake light off"
  - Since it is more natural to think of the labels to be the other way around, utils/detect_brakes.py maps 0->True, 1->False
