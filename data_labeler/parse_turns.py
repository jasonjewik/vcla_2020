import glob
import os
import csv
import math
import numpy as np
import cv2
from natsort import natsorted
import argparse
import vg

from utils import progress_bar

# Set font info for cv2
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 350)
fontScale = 1
fontColor = (0, 255, 143)
thickness = 2

# For writing to image
def write_to_img(img, text):
    result = cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, fontColor, thickness)
    return result

def parse_turns(workspace_path):
    # Get the drive and image data
    drivedata_path = os.path.join(workspace_path, "*.csv")
    imagedata_path = os.path.join(workspace_path, "crop/*.jpg")
    drivedata = natsorted(glob.glob(drivedata_path))
    imagedata = natsorted(glob.glob(imagedata_path))
    
    if len(imagedata) > len(drivedata):
        imagedata = imagedata[:len(drivedata)]
        print('more image data than drive data')
    elif len(drivedata) > len(imagedata):
        drivedata = drivedata[:len(imagedata)]
        print('more drive data than image data')

    # Create results directory if not exist, otherwise clear it
    result_dir = os.path.join(workspace_path, "results")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        print(f'created directory {result_dir}')
    else:
        filelist = os.listdir(result_dir)
        for file in filelist:
            filepath = os.path.join(result_dir, file)
            os.remove(filepath)
        print(f'cleared directory {result_dir}')

    # Parse turns
    headings = []
    positions = []
    two_pi = 2 * np.pi
    idx = 0
    print('parsing turns')

    # Creat file for storing info
    master_data_path = os.path.join(result_dir, 'results.txt')
    master_data = open(master_data_path, 'a')

    for ddata, idata in zip(drivedata, imagedata):
        progress_bar(idx + 1, len(drivedata))

        # Read drive data info
        df = open(ddata, 'r')
        reader = csv.reader(df, delimiter=' ')
        row = next(reader)
        x = float(row[0][1:])
        y = float(row[1])
        z = float(row[2][:-1])
        positions.append(np.array([x, y, z]))
        row = next(reader)
        u = float(row[0][1:])
        v = float(row[1])
        w = float(row[2][:-1])
        headings.append(np.array([u, v, w]))
        df.close()

        # Write turn to the result image
        img = cv2.imread(idata)
        
        if len(headings) >= 2 and len(positions) >= 2:
            # calculates the difference in headings
            prevh = headings[idx - 1]
            currh = headings[idx]
            # returns an angle between pi and -pi
            diffh = vg.signed_angle(prevh, currh, look=vg.basis.z, units='rad')
            diffh /= two_pi

            # calculates the change in distance
            prevp = positions[idx - 1]
            currp = positions[idx]
            # for calculating the euclidean distance in 3-space
            diffp = vg.euclidean_distance(prevp, currp)
            
            parsed_turn = ''
            
            if (diffh >= -0.001 and diffh <= 0.001):
                if (diffp <= 0.001):
                    parsed_turn = 'stopped'
                else:
                    parsed_turn = 'straight'

                # verbose output
                # if (diffp <= 0.001):
                #     parsed_turn = f'stopped, diff: {round(diff, 3)}'
                # else:
                #     parsed_turn = f'straight, diff: {round(diff, 3)}'
            elif (diffh < -0.001):
                parsed_turn = 'right'

                # verbose output
                # parsed_turn = f'right, diff: {round(diff, 3)}'
            elif (diffh > 0.001):
                parsed_turn = 'left'

                # verbose output
                # img = write_to_img(img, f'left, diff: {round(diff, 3)}')

            if parsed_turn == '':
                print('Something terrible has gone wrong')
                exit(1)
            else:
                img = write_to_img(img, parsed_turn)
                master_data.write(f'{idata}: {parsed_turn}\n')

            cv2.imwrite(os.path.join(result_dir, f'{str(idx).zfill(9)}_res.jpg'), img)
            idx += 1

    master_data.close()
    print(f'wrote results to {result_dir}')

if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='Parses turns of the given data'
    )
    parser.add_argument('workspace_path', action='store',
                        help='should be GTAV_program\\drivedata')
    args = parser.parse_args()

    if not os.path.isdir(args.workspace_path):
        print('the specified data folder does not exist')
        print('please gather some data first')
        exit(1)
    
    parse_turns(args.workspace_path)