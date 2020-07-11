# VCLA Autonomous Vehicles in _GTA V_

## Contents

### data_labeler

- `label_data.py`

  - should be the only script you need to run
  - if something goes wrong, try running the following scripts in the order they are shown, except `graph_coords.py`
  - crops images and labels them, putting them into GTAV_program/drivedata/results
    - also generates a file called results.txt in the same folder with all of the parsed turn data
  - if the --parent flag is set, the script expects the following directory structure:

    ```
    ~> /path/to/root_folder/
      |-- child_folder_1
      |   | -- *.csv
      |   | -- RGB_whole.raw
      |-- ...
      |-- child_folder_n
      |   | -- *.csv
      |   | -- RGB_whole.raw
    ```

  - **Sample output**

    ```
    (data_labeler) jason@jason:~/GitRepos/vcla_2020/data_labeler$ python label_data.py $workspace -p
      Detected OS: linux
      found the following child directories:
      /home/jason/Documents/drivedata/2037-070920
      /home/jason/Documents/drivedata/1928-071020
      /home/jason/Documents/drivedata/1457-070320
      /home/jason/Documents/drivedata/2033-070920
      /home/jason/Documents/drivedata/1658-070720
      /home/jason/Documents/drivedata/1651-070720
      Are these the folders you want to use? ([Y]/n)
      generating pickles
      Progress: [ ########## ] 100.0%
      successfully wrote pkl files to /home/jason/Documents/drivedata/2037-070920/pickles
      reading pickle files
      Progress: [ ########## ] 100.0%
      read all pickle files
      generating images
      Progress: [ ########## ] 100.0%
      successfully converted to images, stored in /home/jason/Documents/drivedata/2037-070920/imvid
      successfully cropped images, stored in /home/jason/Documents/drivedata/2037-070920/crop
      more drive data than image data
      cleared directory /home/jason/Documents/drivedata/2037-070920/results
      parsing turns
      Progress: [ ########## ] 100.0%
      wrote results to /home/jason/Documents/drivedata/2037-070920/results
      all data has been labeled
      generating pickles
      Progress: [ ########## ] 100.0%
      successfully wrote pkl files to /home/jason/Documents/drivedata/1928-071020/pickles
      reading pickle files
      Progress: [ ########## ] 100.0%
      read all pickle files
      generating images
      Progress: [ ########## ] 100.0%
      successfully converted to images, stored in /home/jason/Documents/drivedata/1928-071020/imvid
      successfully cropped images, stored in /home/jason/Documents/drivedata/1928-071020/crop
      more drive data than image data
      cleared directory /home/jason/Documents/drivedata/1928-071020/results
      parsing turns
      Progress: [ ########## ] 100.0%
      wrote results to /home/jason/Documents/drivedata/1928-071020/results
      all data has been labeled
      generating pickles
      Progress: [ ########## ] 100.0%
      successfully wrote pkl files to /home/jason/Documents/drivedata/1457-070320/pickles
      reading pickle files
      Progress: [ ########## ] 100.0%
      read all pickle files
      generating images
      Progress: [ ########## ] 100.0%
      successfully converted to images, stored in /home/jason/Documents/drivedata/1457-070320/imvid
      Progress: [ ########## ] 100.0%
      successfully cropped images, stored in /home/jason/Documents/drivedata/1457-070320/crop
      more drive data than image data
      cleared directory /home/jason/Documents/drivedata/1457-070320/results
      parsing turns
      Progress: [ ########## ] 100.0%
      wrote results to /home/jason/Documents/drivedata/1457-070320/results
      all data has been labeled
      generating pickles
      Progress: [ ########## ] 100.0%
      successfully wrote pkl files to /home/jason/Documents/drivedata/2033-070920/pickles
      reading pickle files
      Progress: [ ########## ] 100.0%
      read all pickle files
      generating images
      Progress: [ ########## ] 100.0%
      successfully converted to images, stored in /home/jason/Documents/drivedata/2033-070920/imvid
      successfully cropped images, stored in /home/jason/Documents/drivedata/2033-070920/crop
      more drive data than image data
      cleared directory /home/jason/Documents/drivedata/2033-070920/results
      parsing turns
      Progress: [ ########## ] 100.0%
      wrote results to /home/jason/Documents/drivedata/2033-070920/results
      all data has been labeled
      generating pickles
      Progress: [ ########## ] 100.0%
      successfully wrote pkl files to /home/jason/Documents/drivedata/1658-070720/pickles
      reading pickle files
      Progress: [ ########## ] 100.0%
      read all pickle files
      generating images
      Progress: [ ########## ] 100.0%
      successfully converted to images, stored in /home/jason/Documents/drivedata/1658-070720/imvid
      successfully cropped images, stored in /home/jason/Documents/drivedata/1658-070720/crop
      more drive data than image data
      cleared directory /home/jason/Documents/drivedata/1658-070720/results
      parsing turns
      Progress: [ ########## ] 100.0%
      wrote results to /home/jason/Documents/drivedata/1658-070720/results
      all data has been labeled
      generating pickles
      Progress: [ ########## ] 100.0%
      successfully wrote pkl files to /home/jason/Documents/drivedata/1651-070720/pickles
      reading pickle files
      Progress: [ ########## ] 100.0%
      read all pickle files
      generating images
      Progress: [ ########## ] 100.0%
      successfully converted to images, stored in /home/jason/Documents/drivedata/1651-070720/imvid
      successfully cropped images, stored in /home/jason/Documents/drivedata/1651-070720/crop
      more drive data than image data
      cleared directory /home/jason/Documents/drivedata/1651-070720/results
      parsing turns
      Progress: [ ########## ] 100.0%
      wrote results to /home/jason/Documents/drivedata/1651-070720/results
      all data has been labeled
    ```

- `generate_pickles.py`

  - creates .pkl files from GTAV_program/drivedata/RGB_whole.raw
  - puts them into GTAV_program/drivedata/pickles
  - can take a long time before the progress bar is shown if the RGB_whole.raw file is very large

- `generate_visuals.py`

  - creates .jpg and/or .avi files from GTAV_program/drivedata/pickles/\*.pkl
  - puts them into GTAV_program/drivedata/imvid

- `crop_images.py`

  - crops the .jpg files in GTAV_program/drivedata/pickles/\*.pkl
  - puts them into GTAV_program/drivedata/crop
  - assumes a motorcycle as viewed in 3rd person mode

- `parse_turns.py`

  - parse the turns based on data in GTAV_program/drivedata/\*.csv and the .jpg files in GTAV_program/drivedata/crop
  - determined by measuring the difference in the vehicle's headings between two points

- `graph_coords.py`

  - displays a graphical representation of the coordinates in GTAV_program/drivedata/\*.csv

## To Do

[x] Parse drivedata CSV files
[ ] Train YOLOv3 model to determine braking/not-braking

## Further Notes

- YOLOv3 model trained with [Darknet](https://github.com/pjreddie/darknet)
- Image annotations done with [LabelImg](https://github.com/tzutalin/labelImg)
