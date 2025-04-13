import vt
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import skimage.filters as fl
import os
import csv
from scipy.interpolate import interp1d

# Load main image
img = imread("28C.tif")

dt = 0.2
n_plots = 5
step = 3

# Height ranges for 11 files
height_ranges = {
        1: (59, 130),
        2: (55, 127),
        3: (77, 128),
        4: (78, 121),
        5: (49, 108),
        6: (32, 92),
        7: (51, 110),
        8: (38, 97),
        9: (55, 107),
    }

def transform(img, file_index):
    h_min, h_max = height_ranges.get(file_index, (60, 140))
    return img[h_min:h_max, :]

# Collect all velocity data
all_velocities = {}
file_list = sorted(os.listdir("input_28"))[:11]

for idx, file in enumerate(file_list, start=1):
    img = imread(f"input_28/{file}")
    m = {}
    for i in range(0, step * n_plots, step):  
        m[i] = []
    
    for i in range(0, step * n_plots, step):  
        img_float, img_ref = img[i], img[i + step]
        img_float = transform(img_float, idx)
        img_ref = transform(img_ref, idx)
        
        img_float_ = vt.vtImage(img_float)
        img_ref_ = vt.vtImage(img_ref)

        trnsf = vt.blockmatching(img_float_, image_ref=img_ref_, params="-transformation-type vectorfield -no-verbose")
        
        nx = img_float.shape[1] // 5
        ny = img_float.shape[0] // 5
        X, Y = np.meshgrid(np.linspace(0, img_float.shape[1], nx), np.linspace(0, img_float.shape[0], ny))
        points = [(i, j) for i, j in zip(X.flatten(), Y.flatten())]
        points_ = vt.vtPointList(points)

        points_transformed_ = vt.apply_trsf_to_points(points_, trnsf)
        points_transformed = points_transformed_.copy_to_array()[:, :2]
        vectorx = (-points_transformed[:, 0].reshape(ny, nx) + X) / (dt * step)
        vectory = (-points_transformed[:, 1].reshape(ny, nx) + Y) / (dt * step)
        vectorx[np.abs(vectorx) < 0.00002] = 0
        vectory[np.abs(vectory) < 0.00002] = 0
        mean = vectorx.mean(axis=0)
        m[i].append(mean - mean[::-1])
    
    # Find velocities at specified microns using interpolation (without normalization)
    micron_positions = [14, 14.5, 15, 15.5, 16, 16.5]
    velocities = {}
    for i, j in m.items():
        mean_curve = np.mean(j, axis=0)
        x_existing = np.arange(len(mean_curve))
        interp_func = interp1d(x_existing, mean_curve, kind='linear', fill_value='extrapolate')
        velocities[i] = [interp_func(pos) for pos in micron_positions]
    
    all_velocities[file] = velocities


# Write all velocity data into a single CSV file with time step for each file in a separate row
csv_filename = "all_velocities.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    
    # Sort time steps and iterate over them
    time_steps = sorted(set(time for file in all_velocities for time in all_velocities[file]))
    
    for time in time_steps:
        for file in file_list:
            row = [(time+1) * dt ]  # Time step for this file
            row.extend(all_velocities[file].get(time, [0] * len(micron_positions)))  # Corresponding velocities
            writer.writerow(row)

print(f"Saved all velocity data in {csv_filename}")


