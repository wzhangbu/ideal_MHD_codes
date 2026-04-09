#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processes Paraview CSV trace exports. Applies step-function scanning 
to identify termination boundaries, convert coordinates (GSM to SM), 
and plot termination point overlays.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from spacepy import coordinates as coord
from spacepy.time import Ticktock

# --- Configuration ---
EVENT_INDEX = 2
HIGH_RESOLUTION = 0
EVENT_LIST = ['20140508', '20140417', '20181020']

if EVENT_INDEX == 2:
    figure_list = ['run12_hall', 'run11_ideal', 'run22_ideal', 'run23_hall', 
                   'run24_hall', 'run25_ideal', 'run26_hall', 'run27_epic', 
                   'run31_epic', 'run63_epic', 'run80_epic', 'run81_epic']
    figure_index = 5
    base_dir = f'/Users/weizhang/Desktop/research/EMIC/{EVENT_LIST[EVENT_INDEX]}/{figure_list[figure_index]}'
    path = os.path.join(base_dir, 'keogram_MLAT_1/' if HIGH_RESOLUTION == 1 else 'keogram_MLAT/')

def Get_MLT(x, y):
    """Calculates MLT based on equatorial X and Y plane coordinates."""
    mlt = np.arctan2(y, x) / np.pi * 180 / 15 + 12
    return np.mod(mlt, 24)

def read_data(file):
    """Loads CSV exported from Paraview containing Magnetic data and trace properties."""
    data_all = np.loadtxt(file, dtype=float, comments='#', skiprows=1, delimiter=',')
    B = data_all[:, 0:3]
    Termination = data_all[:, 3:4]
    X = data_all[:, 4:7]
    MLT_info = Get_MLT(X[:, 0], X[:, 1]).reshape(-1, 1)
    return B, Termination, X, MLT_info    

def Get_MLAT(X):
    """Derives Magnetic Latitude based on geometry rules."""
    return np.arctan(X[2] / np.sqrt(X[0]**2 + X[1]**2)) / np.pi * 180 

def Step_function(termination):
    """
    Minimizes a step function constraint looking for the optimal 
    cut-off point where termination switches from closed (1) to open (5).
    """
    y1, y2 = 1, 5
    sum_min = 1e10
    index = np.nan
    
    for i in range(len(termination)):
        tmp = np.sum(np.abs(termination[0:i] - y1)) + np.sum(np.abs(termination[i:-1] - y2))
        if tmp < sum_min:
            sum_min = tmp
            index = i
            
    return index

def Get_OCB(X, Termination, MLT_info, yrange, MLT_num):
    """Loops through unique MLT slices to evaluate step-functions and pull accurate boundaries."""
    MLT_info = np.around(MLT_info, 2)
    MLT_unique = np.unique(MLT_info)
    
    OCB_MLAT = np.zeros(MLT_num)
    OCB_XYZ = np.zeros((MLT_num, 3))
    
    unit_num = len(np.argwhere(abs(MLT_info - MLT_unique[0]) < 0.001)[:, 0])
    k = 0
    
    for i in range(len(MLT_unique)):
        if (MLT_unique[i] <= yrange[0]) or (MLT_unique[i] >= yrange[1]):
            continue
            
        tmp = np.argwhere(abs(MLT_info - MLT_unique[i]) < 0.001)[:, 0]
        if len(tmp) < unit_num:
            continue
            
        term_mlt = Termination[tmp]
        X_mlt = X[tmp, :]
        loca = Step_function(term_mlt)
        
        if not np.isnan(loca):
            OCB_MLAT[k] = Get_MLAT(X_mlt[loca, :])
            OCB_XYZ[k, :] = X_mlt[loca, :]
        else: 
            OCB_MLAT[k] = 65
            
        if EVENT_INDEX == 2 and OCB_MLAT[k] < 69:
            OCB_MLAT[k] = np.nan
            OCB_XYZ[k, :] = np.nan
        elif EVENT_INDEX == 1 and OCB_MLAT[k] < 30:
            OCB_MLAT[k] = np.nan
            OCB_XYZ[k, :] = np.nan
            
        k += 1
        
    return OCB_XYZ, OCB_MLAT

def GSM2SM(X, time, car=1):
    """Converts coordinate bases via Spacepy. Ticktock initializes exact timesteps."""
    X_GSM = coord.Coords(X, 'GSM', 'car')
    times = np.repeat(time, len(X[:, 0]), axis=0)
    X_GSM.ticks = Ticktock(times)
    
    X_out = X_GSM.convert('SM', 'car' if car == 1 else 'sph')
    return X_out.data

def Plot_dots(X, OCB, Termination, vnorm=0, axis=[0, 1], filenames='filenames.cvs'):
    """Scatter plot visualization mapping Termination status mapping via coolwarm scales."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    norm = mcolors.Normalize(vmin=vnorm[0], vmax=vnorm[1]) if vnorm != 0 else mcolors.Normalize(vmin=np.nanmin(Termination), vmax=np.nanmax(Termination))
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Termination, cmap='bwr')
    
    if np.nanmax(OCB) != -1:
        ax.scatter(OCB[:, axis[0]], OCB[:, axis[1]], c='green')
        
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.text(0.1, 0.9, f"time = {filenames}", size=15)
    
    MLAT = [65, 70, 75, 80, 85] if EVENT_INDEX == 1 else [70, 75, 80, 85]
    for mlat in MLAT:
        phi = np.arange(361)
        xy_length = 3 * np.cos(np.pi * mlat / 180.)
        x = xy_length * np.cos(np.pi * phi / 180.)
        y = xy_length * np.sin(np.pi * phi / 180.)
        ax.plot(x, y, color='grey')
        
    pts = 25 if EVENT_INDEX == 1 else 20
    x_lines = np.linspace(0, 1.5, pts)
    ax.plot(x_lines, np.linspace(0, -0.5, pts), color='orange')
    ax.plot(x_lines, np.linspace(0, 0.35, pts), color='orange')   
    ax.plot(x_lines, np.linspace(0, 2.6, pts), color='orange')  
    
    plt.show()

# --- Execution ---
if __name__ == "__main__":
    os.chdir(path)  
    files = sorted(glob.glob('./*.csv'))
    print(f'There are {len(files)} files in total.')

    SM = 0
    MLT_set = 1
    yrange = [6, 18] if MLT_set == 1 else [-12, 12]

    # Precalculate shapes dynamically off the first frame
    _, _, _, MLT_info_init = read_data(files[0])
    MLT_unique = np.unique(np.around(MLT_info_init, 2))
    valid_mask = (MLT_unique <= yrange[1]) & (MLT_unique >= yrange[0])
    MLT_num = len(np.argwhere(valid_mask)[:, 0])
    MLT_unique_output = MLT_unique[valid_mask]

    MLAT_keogram_GSM = np.zeros((MLT_num, len(files)))
    if SM == 1:
        OCB_MLT_SM = np.zeros((MLT_num, len(files)))
        OCB_MLAT_SM = np.zeros((MLT_num, len(files)))

    for k, file in enumerate(files):
        print(file)
        B, Termination, X, MLT_info = read_data(file)
        OCB_XYZ, OCB_MLAT = Get_OCB(X, Termination, MLT_info, yrange, MLT_num)
        
        time_str = f'2018-10-20T21:{k + 25}' if EVENT_INDEX == 2 else f'2014-04-17T17:{k + 25}'
        
        X_SM = GSM2SM(X, time_str, car=1)    
        OCB_SM = GSM2SM(OCB_XYZ, time_str, car=1) 
        OCB_SM_sph = GSM2SM(OCB_XYZ, time_str, car=0)
        
        MLAT_keogram_GSM[:, k] = OCB_SM_sph[:, 1]
        
        if SM == 1:
            OCB_MLT_SM[:, k] = Get_MLT(OCB_SM[:, 0], OCB_SM[:, 1])
            OCB_MLAT_SM[:, k] = OCB_SM_sph[:, 1]

        Plot_dots(X_SM, OCB_SM, Termination, vnorm=0, filenames=f"{file[2:4]}_SM")
        
    # Write data vectors to file
    if SM == 0:
        np.savetxt(os.path.join(path, f"{figure_list[figure_index]}_MLT_info_618.txt"), MLT_unique_output, fmt='%.6f', delimiter=',')
        np.savetxt(os.path.join(path, f"{figure_list[figure_index]}_MLAT_info_618.txt"), MLAT_keogram_GSM, fmt='%.6f', delimiter=',')
    else:
        np.savetxt(os.path.join(path, f"{figure_list[figure_index]}_SM_MLT_info_618.txt"), OCB_MLT_SM, fmt='%.6f', delimiter=',')
        np.savetxt(os.path.join(path, f"{figure_list[figure_index]}_SM_MLAT_info_618.txt"), OCB_MLAT_SM, fmt='%.6f', delimiter=',')