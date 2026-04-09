#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Magnetosphere OCB Post-Processing.
Reads Paraview tracing data to determine field line termination status 
(Closed vs Open) and extracts boundary coordinates via Step Function fitting.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from spacepy import coordinates as coord
from spacepy.time import Ticktock

def Get_MLT(x, y):
    """Calculates MLT based on Cartesian coordinates."""
    mlt = np.arctan2(y, x) / np.pi * 180 / 15 + 12
    return np.mod(mlt, 24)

def read_data(file):
    """Reads trace CSVs generated from ParaView."""
    data_all = np.loadtxt(file, dtype=float, comments='#', skiprows=1, delimiter=',')
    B = data_all[:, 0:3]
    Termination = data_all[:, 3:4]
    X = data_all[:, 4:7]
    MLT_info = Get_MLT(X[:, 0], X[:, 1]).reshape(-1, 1)
    return B, Termination, X, MLT_info    

def Get_MLAT(X):
    """Calculates MLAT from Cartesian array."""
    return np.arctan(X[2] / np.sqrt(X[0]**2 + X[1]**2)) / np.pi * 180 

def Step_function(termination):
    """
    Minimizes error against an ideal step function to pinpoint the exact index 
    where field lines transition from Closed (Label: 1) to Open (Label: 5).
    """
    y1, y2 = 1, 5
    sum_min = 1e10
    index = np.nan
    
    for i in range(len(termination)):
        # Calculate error of current split point
        tmp = np.sum(np.abs(termination[0:i] - y1)) + np.sum(np.abs(termination[i:] - y2))
        if tmp < sum_min:
            sum_min = tmp
            index = i
            
    return index

def Get_OCB(X, Termination, MLT_info, yrange, MLT_num, event_index=2):
    """Isolates Open-Closed Boundary coordinates by analyzing each MLT longitudinal bin."""
    MLT_info = np.around(MLT_info, 2)
    MLT_unique = np.unique(MLT_info)
    
    OCB_MLAT = np.zeros(MLT_num)
    OCB_XYZ = np.zeros((MLT_num, 3))
    k = 0
    unit_num = len(np.argwhere(abs(MLT_info - MLT_unique[0]) < 0.001)[:, 0])
    
    for i in range(len(MLT_unique)):
        # Filter bins outside of requested range
        if (MLT_unique[i] <= yrange[0]) or (MLT_unique[i] >= yrange[1]):
            continue
            
        tmp = np.argwhere(abs(MLT_info - MLT_unique[i]) < 0.001)[:, 0]
        if len(tmp) < unit_num:
            continue
            
        term_mlt = Termination[tmp]
        X_mlt = X[tmp, :]
        
        # Apply the step function to find the transition index
        loca = Step_function(term_mlt)
        
        if not np.isnan(loca):
            OCB_MLAT[k] = Get_MLAT(X_mlt[loca, :])
            OCB_XYZ[k, :] = X_mlt[loca, :]
            
            # Apply safety bounds based on event configuration
            if event_index == 1 and OCB_MLAT[k] < 30:
                OCB_MLAT[k], OCB_XYZ[k, :] = np.nan, np.nan
        k += 1
        
    return OCB_XYZ, OCB_MLAT

def GSM2SM(X, time, car=1):
    """Converts arrays from GSM to SM using spacepy Ticktock."""
    X_GSM = coord.Coords(X, 'GSM', 'car')
    times = np.repeat(time, len(X[:, 0]), axis=0)
    X_GSM.ticks = Ticktock(times)
    X_out = X_GSM.convert('SM', 'car' if car == 1 else 'sph')
    return X_out.data

def Plot_dots(X, OCB, Termination, Re, vnorm=0, filenames='filenames.csv'):
    """Scatter plot representation of field line termination topology."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    norm = mcolors.Normalize(vmin=vnorm[0], vmax=vnorm[1]) if vnorm != 0 else mcolors.Normalize(vmin=np.nanmin(Termination), vmax=np.nanmax(Termination))
    
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Termination, cmap='bwr')
    if np.nanmax(OCB) != -1:
        ax.scatter(OCB[:, 0], OCB[:, 1], c='green')  # Overlay identified boundary in Green
        
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.text(0.1, 0.9, f"time = {filenames[2:4]}", size=15)
    
    if Re == 9.5:
        ax.set_xlim([-7.5, 7.5])
        ax.set_ylim([-7.5, 7.5])

    plt.show()

if __name__ == "__main__":
    EVENT_INDEX = 2
    FIGURE_INDEX = 5
    SM = 1
    RE = 2.7
    YRANGE = [6, 18]

    FIGURE_LIST = ['run12_hall', 'run11_ideal', 'run22_ideal', 'run23_hall', 
                   'run24_hall', 'run25_ideal', 'run26_hall', 'run27_epic', 
                   'run31_epic', 'run63_epic', 'run80_epic', 'run81_epic']

    # Path Configuration Logic
    BASE_PATH = f'/Users/weizhang/Desktop/research/EMIC/20181020/{FIGURE_LIST[FIGURE_INDEX]}'
    if RE == 3: PATH = f"{BASE_PATH}/keogram_MLAT_3RE/"
    elif RE == 2.7: PATH = f"{BASE_PATH}/keogram_MLAT_2.7RE/"
    elif RE == 9.5: PATH = f"{BASE_PATH}/RE_Uperp/9.5RE_OCB/"
    else: PATH = f"{BASE_PATH}/keogram_MLAT_4RE/"
    
    os.chdir(PATH)  
    files = sorted(glob.glob('./*.csv'))
    print(f'Found {len(files)} PV trace files in total.')

    # Pre-calculate bin arrays based on the first file's geometry
    _, _, _, MLT_info = read_data(files[0])
    MLT_unique = np.unique(np.around(MLT_info, 2))
    
    valid_mask = (MLT_unique <= YRANGE[1]) & (MLT_unique >= YRANGE[0])
    MLT_num = np.sum(valid_mask)
    MLT_unique_output = MLT_unique[valid_mask]

    MLAT_keogram_GSM = np.zeros((MLT_num, len(files)))
    if SM == 1:
        OCB_MLT_SM = np.zeros((MLT_num, len(files)))
        OCB_MLAT_SM = np.zeros((MLT_num, len(files)))

    # Process all temporal frames
    for k, file in enumerate(files):
        B, Termination, X, MLT_info = read_data(file)
        OCB_XYZ, OCB_MLAT = Get_OCB(X, Termination, MLT_info, YRANGE, MLT_num, event_index=EVENT_INDEX)
        
        time_str = f'2018-10-20T21:{k+25}' if EVENT_INDEX == 2 else f'2014-04-17T17:{k+25}'
        MLAT_keogram_GSM[:, k] = OCB_MLAT
        
        # Apply Coordinate Transforms
        X_SM = GSM2SM(X, time_str, car=1)    
        OCB_SM = GSM2SM(OCB_XYZ, time_str, car=1) 
        OCB_SM_sph = GSM2SM(OCB_XYZ, time_str, car=0)
        
        if SM == 1:
            OCB_MLT_SM[:, k] = Get_MLT(OCB_SM[:, 0], OCB_SM[:, 1])
            OCB_MLAT_SM[:, k] = OCB_SM_sph[:, 1]

        Plot_dots(X_SM, OCB_SM, Termination, Re=RE, vnorm=0, filenames=file)
        
    # Save processed boundary arrays
    prefix = os.path.join(PATH, FIGURE_LIST[FIGURE_INDEX])
    if SM == 0:
        np.savetxt(f"{prefix}_GSM_MLT_info_618_{RE}RE.txt", MLT_unique_output, fmt='%.6f', delimiter=',')
        np.savetxt(f"{prefix}_GSM_MLAT_info_618_{RE}RE.txt", MLAT_keogram_GSM, fmt='%.6f', delimiter=',')
    elif SM == 1:
        np.savetxt(f"{prefix}_SM_MLT_info_618_{RE}RE.txt", OCB_MLT_SM, fmt='%.6f', delimiter=',')
        np.savetxt(f"{prefix}_SM_MLAT_info_618_{RE}RE.txt", OCB_MLAT_SM, fmt='%.6f', delimiter=',')