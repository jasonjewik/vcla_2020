import os
import csv
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Generates a 3D graph of the given drivedata'
)
parser.add_argument('data_path', action='store',
                    help='should be GTAV_program/drivedata')
args = parser.parse_args()

if not os.path.isdir(args.data_path):
    print('the specified data folder does not exist')
    print('please gather some data first')
    exit(1)

data_path = args.data_path
files = os.listdir(data_path)[:50]

x_coords = []
y_coords = []
z_coords = []
u = []
v = []
w = []

for file in files:
    if file[-3:] == 'csv':
        fpath = os.path.join(data_path, file)
        with open(fpath, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            row = next(reader)
            x = float(row[0][1:])
            y = float(row[1])
            z = float(row[2][:-1])
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            row = next(reader)
            u.append(float(row[0][1:]))
            v.append(float(row[1]))
            w.append(float(row[2][:-1]))

fig = plt.figure()
axes = fig.gca(projection='3d')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('z')

line = axes.plot3D(x_coords, y_coords, z_coords, 'green')
vecs = axes.quiver(x_coords, y_coords, z_coords, u,
                   v, w, length=5, normalize=True)

plt.show()
