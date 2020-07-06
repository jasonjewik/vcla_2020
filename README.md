# VCLA Autonomous Vehicles in *GTA V*

## Contents
### data_labeler
- `generate_pickles.py`
  - creates .pkl files from GTAV\_program\drivedata\RGB\_whole.raw
  - puts them into GTAV\_program\drivedata\pickles

- `generate_visuals.py`
  - creates .jpg and/or .avi files from GTAV\_program\drivedata\pickles\\*.pkl
  - puts them into GTAV_program\drivedata\imvid

- `graph_coords.py`
  - displays a graphical representation of the coordinates in GTAV_program\drivedata\\*.csv

- `crop_images.py`
  - crops the .jpeg files in GTAV\_program\drivedata\pickles\\*.pkl
  - puts them into GTAV\_program\drivedata\crop
  - assumes a motorcycle as viewed in 3rd person mode

## Further Notes
- Image annotations created with [LabelImg, Windows_v1.8.0](https://tzutalin.github.io/labelImg/)