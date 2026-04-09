#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radial Boundary Extractor.
Parses CSV tracing records to isolate the specific MLAT and MLT coordinates 
of the magnetospheric boundary where the radius strictly matches a target value (e.g., 9.5 Re).
"""

import numpy as np
import os
import glob
from read_data import Data 
from spacepy import coordinates as coord
from spacepy.time import Ticktock

def GSM2SM(X, time, car=1):
    """Spacepy implementation for GSM to SM coordinate conversion."""
    X_temp = X.reshape(-1, 3)
    X_GSM = coord.Coords(X_temp, 'GSM', 'car')
    times = np.repeat(time, len(X_temp[:, 0]), axis=0)
    X_GSM.ticks = Ticktock(times)
    
    if car == 1:
        X_out = X_GSM.convert('SM', 'car')
    else:
        X_out = X_GSM.convert('SM', 'sph')
    
    return X_out.data.reshape(X.shape)

def getMLT(x, y):
    """Calculates Magnetic Local Time (MLT) wrapping from 0 to 24."""
    mlt = np.arctan2(y, x) / np.pi * 180 / 15 + 12
    return np.mod(mlt, 24)

if __name__ == "__main__":
    
    # Target constraint radius (in Earth Radii)
    SPHERE_RE = 9.5
    
    PATH = '/Users/weizhang/Desktop/research/EMIC/20181020/run25_ideal/RE_Uperp/tracing_data/'
    os.chdir(PATH)
    
    # Load dawn and dusk trace files sequentially
    fs1 = sorted(glob.glob('./T*dawn.csv'))
    fs2 = sorted(glob.glob('./T*dusk.csv'))
    
    print(f'Found {len(fs1)} time-length data frames.')
    
    # Initialize arrays for MLAT, MLT, and Cartesian XYZ outputs
    OCB_MLAT_9RE = np.zeros((len(fs1), 2)) * np.nan
    OCB_MLT_9RE = np.zeros((len(fs1), 2)) * np.nan
    OCB_XYZ = np.zeros((len(fs1), 6)) * np.nan
    
    fs = [fs1, fs2]
    
    # Iterate through all files (timeframes)
    for j in range(len(fs1)):
        # i=0 corresponds to Dawn, i=1 corresponds to Dusk
        for i in range(2):
            results = Data()
            results = Data._read_from_file_ideal(results, os.path.join(PATH, fs[i][j]))
            
            # Fetch array column indices based on headers
            re_index = results.names.index('re')
            ix = results.names.index('x')
            z_index = results.names.index('z')
            mlat_index = results.names.index('mlat')
            mlt_index = results.names.index('mlt')
            
            # Sub-select points that strictly lie within the specified radial shell (e.g. 9.5 Re +/- 0.03 Re)
            # AND reside in the northern hemisphere (Z > 0)
            tmp = np.argwhere((SPHERE_RE - 0.03 < results.data[..., re_index]) &
                              (SPHERE_RE + 0.03 > results.data[..., re_index]) &
                              (results.data[..., z_index] > 0))[:, 0]
                              
            if len(tmp) == 0:
                print(f"Warning: No valid points found at {SPHERE_RE} Re for file {fs[i][j]}")
                continue
                
            # Extract median attributes of the boundary shell slice to stabilize against noise
            OCB_MLAT_9RE[j, i] = np.median(results.data[tmp, mlat_index])
            OCB_MLT_9RE[j, i] = np.median(results.data[tmp, mlt_index])
            
            # Extract average XYZ geometry and convert to SM coordinates
            x1 = np.nanmean(results.data[tmp, ix])
            y1 = np.nanmean(results.data[tmp, ix+1])
            z1 = np.nanmean(results.data[tmp, ix+2])
            temp1 = np.array((x1, y1, z1))
            
            sm_coord = GSM2SM(temp1, '2018-10-20T21:40', car=1)
            OCB_XYZ[j, i*3 : i*3+3] = sm_coord
            
            # Re-verify the MLT based strictly on the SM converted geometry
            OCB_MLT_9RE[j, i] = getMLT(OCB_XYZ[j, i*3], OCB_XYZ[j, i*3+1])
            
    print("Extracted MLT Bounds:\n", OCB_MLT_9RE)
    
    # Save the boundaries to text file for the next pipeline step
    out_dir = f'/Users/weizhang/Desktop/research/EMIC/20181020/run25_ideal/RE_Uperp/{SPHERE_RE}RE/'
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, f'GM2{SPHERE_RE}RESM.txt'), OCB_MLT_9RE)