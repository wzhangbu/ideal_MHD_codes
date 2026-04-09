#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for processing Ionosphere Electrodynamics (IE) and OCB data.
Provides functions for coordinate transformations and the KeogramData class.
"""

import os
import sys
import glob
import numpy as np
import spacepy.pybats.rim as rim

def get_IE_OCB_data(file_path):
    """
    Reads IE OCB data from a text file with varying column lengths.
    Pads missing data with NaNs.
    """
    max_time_num = 0
    k = 0
    with open(file_path, 'r') as f:
        data = f.readlines()
        for line in data:
            odom = line.split()
            if len(odom) > max_time_num:
                max_time_num = len(odom)
            k += 1
    
    a = np.zeros((k, max_time_num)) * np.nan
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(data):
            odom = line.split()
            a[i, 0:len(odom)] = np.array(odom, dtype=float)
            
    return a

def fit_OCB(MLT_IE, MLAT_IE):
    """
    Fits a quadratic polynomial to the OCB MLAT vs MLT data to smooth the boundary.
    """
    valid = ~(np.isnan(MLT_IE) | np.isnan(MLAT_IE))
    coef = np.polyfit(MLT_IE[valid], MLAT_IE[valid], 2)
    y_fit = np.polyval(coef, MLT_IE[valid])
    
    mlt = MLT_IE[valid]
    mlat = y_fit
    return mlt, mlat

def MLAT2XYZ(MLT, MLAT, Re=1.01):
    """
    Converts from Magnetic Local Time (MLT) and Magnetic Latitude (MLAT) 
    to Cartesian coordinates (XYZ) in Solar Magnetic (SM) system.
    """
    num = len(MLT)
    results = np.zeros((num, 3)) * np.nan
    
    for i in range(num):
        angle = (MLT[i] - 12) * 15 * np.pi / 180.
        x = 1.0 if (6 <= MLT[i] <= 18) else -1.0
        y = np.tan(angle) * x
        z = np.sqrt(1 + y**2) * np.tan(np.pi * MLAT[i] / 180.)
        r = np.sqrt(1 + y**2 + z**2)
        
        results[i, 0] = (x / r) * Re
        results[i, 1] = (y / r) * Re
        results[i, 2] = (z / r) * Re
        
    return results

class KeogramData:
    """
    Class to store and process Keogram data from BATSRUS IE files.
    """
    def __init__(self, data=None, plot_ux=None):
        self.data = data
        self.__file_num = 0
        self.__wide_range = 1
        self.__start_index = 0
        self.MLT_cal_range = [6, 18]
        self.__IE_start = 2125
        self.__IE_end = 2150
        self.__PLOT_UX = plot_ux

    def read_from_file(self, path, file_num, start_index, run_name, wide_range, plot_ux=None, MLT_range=[6, 18]):
        """
        Reads IE files across a given time series to construct a Keogram MLT matrix.
        """
        self.__file_num = file_num
        self.__start_index = start_index
        self.__wide_range = wide_range
        self.MLT_cal_range = MLT_range
        self.__PLOT_UX = plot_ux
        
        if self.__wide_range == 1:
            MLAT_start = 76
            MLAT_end = 82
            if run_name == 'run25_ideal':
                MLAT_start, MLAT_end = 76, 82
        else:
            IE_start = 2125 if run_name == 'run25_ideal' else self.__IE_start
            IE_end = 2150 if run_name == 'run25_ideal' else self.__IE_end
            
            IE_MLT_file = os.path.join(path, f"{run_name[0:5]}_IE_MLT_618.txt")
            IE_MLAT_file = os.path.join(path, f"{run_name[0:5]}_IE_MLAT_618.txt")
            
            if run_name == 'run81_epic':    
                IE_MLT_file = os.path.join(path, f"{run_name[0:5]}_GSM_IE_MLT_618_3RE.txt")
                IE_MLAT_file = os.path.join(path, f"{run_name[0:5]}_GSM_IE_MLAT_618_3RE.txt")
            
            print(f"Loading OCB File: {IE_MLT_file}")
            MLT_IE_all = get_IE_OCB_data(IE_MLT_file)
            MLAT_IE_all = get_IE_OCB_data(IE_MLAT_file)
        
        keogram_mlt_num = int(360 / 24 * (self.MLT_cal_range[1] - self.MLT_cal_range[0]))
        keogram_mlt = np.zeros((keogram_mlt_num, file_num))
        
        os.chdir(path)  
        files = sorted(glob.glob('./it*.idl'))
        print(f'Found {len(files)} IDL files in total.')
        
        for k, filename in enumerate(files[start_index : start_index + file_num]):
            print(f"Processing: {filename}")
            ie = rim.Iono(filename)
            time = int(ie.meta['file'].split('/')[-1][9:13])
            
            if wide_range == 0:
                if time < IE_start:
                    MLT_IE, MLAT_IE = MLT_IE_all[:, 0], MLAT_IE_all[:, 0]
                elif time > IE_end:
                    MLT_IE, MLAT_IE = MLT_IE_all[:, -1], MLAT_IE_all[:, -1]
                else:
                    MLT_IE, MLAT_IE = MLT_IE_all[:, time - IE_start], MLAT_IE_all[:, time - IE_start]
                
                MLT_IE, MLAT_IE = fit_OCB(MLT_IE, MLAT_IE)
            
            # Calculate Poleward Velocity (2-D structure)
            ie["n_r"] = np.sqrt(ie["n_x"]**2 + ie["n_y"]**2)
            ie["n_pole_Yuxi"] = - (ie["n_ux"] * ie["n_x"] + ie["n_uy"] * ie["n_y"]) / ie["n_r"]
            n_upole = np.array(ie["n_pole_Yuxi"]) * 1000  # km to m
            
            if plot_ux == "n_ux":
                n_upole = -1 * ie[plot_ux] * 1000
            elif plot_ux is not None:
                n_upole = ie[plot_ux]
        
            # Determine MLT from X and Y coordinates
            mlt = np.arctan2(ie["n_y"], ie["n_x"]) / np.pi * 180 / 15 + 12
            mlt = np.squeeze(np.mod(mlt, 24))
            latitude = np.array(ie["n_theta"])[:, 0]
            
            # Filter by MLT range
            mlt_range_idx = np.squeeze(np.where((mlt[0,:] >= self.MLT_cal_range[0]) & (mlt[0,:] <= self.MLT_cal_range[1])))
            mlt_range_sort = np.squeeze(np.argsort(mlt[0, mlt_range_idx]))
            
            max_upole_mlt = np.zeros(keogram_mlt_num) * np.nan
            
            for i in range(keogram_mlt_num):
                if wide_range == 0:
                    mlt_now = mlt[0, mlt_range_idx[mlt_range_sort[i]]]
                    tmp = np.nanargmin(np.abs(mlt_now - MLT_IE))
                    mlat_now = MLAT_IE[tmp]
                    latitude_range = np.where((latitude >= 90 - mlat_now - 0.6) & (latitude <= 90 - mlat_now + 0.6))
                else:
                    latitude_range = np.where((latitude >= 90 - MLAT_end) & (latitude <= 90 - MLAT_start))                  

                if plot_ux is not None:
                    temp1 = np.nanargmax(np.abs(n_upole[latitude_range, mlt_range_idx[mlt_range_sort[i]]]))
                    temp = n_upole[latitude_range[0][temp1], mlt_range_idx[mlt_range_sort[i]]]
                else:
                    temp = n_upole[latitude_range, mlt_range_idx[mlt_range_sort[i]]].max(axis=1)
                    
                max_upole_mlt[i] = temp
                
            keogram_mlt[:, k] = max_upole_mlt
            
            # Apply thresholds for specific time window
            if time < 2140:
                a = keogram_mlt[0: 20, k]
                a[a > 150] = 150
                keogram_mlt[0: 20, k] = a

        self.data = keogram_mlt
        return self

if __name__ == "__main__":
    # --- CONFIGURATION ---
    run_name = 'run25_ideal'
    base_dir = '/Users/weizhang/Desktop/research/EMIC/20181020/'
    # ---------------------
    
    if run_name == 'run25_ideal':
        file_num, start_index, wide_range, plot_ux = 59, 0, 0, "n_ux"
    elif run_name == 'run26_hall':
        file_num, start_index, wide_range, plot_ux = 59, 0, 0, None
    elif run_name == 'run31_epic':
        file_num, start_index, wide_range, plot_ux = 59, 0, 0, None
        
    path = os.path.join(base_dir, f'{run_name}/IE/')
    keogram_data = KeogramData()
    keogram_data.read_from_file(path, file_num, start_index, run_name, wide_range, plot_ux)