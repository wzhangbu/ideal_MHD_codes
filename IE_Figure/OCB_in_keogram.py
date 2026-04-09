#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads keogram boundaries and cross-references IE MLAT/MLT data 
to extract exact bounding coordinates.
"""

import numpy as np
import os

def load_boundary(file_path, start_time, timelength):
    """
    Loads text boundary limits for a specified temporal window.
    """
    a = np.loadtxt(file_path)    
    extracted_slice = a[start_time : start_time + timelength, :]
    print(f"Loaded boundary shape: {extracted_slice.shape}")
    return extracted_slice

def load_OCB_all_info(boundary, MLT_file, MLAT_file, IE_OCB_start_index, timelength):
    """
    Loads complete MLT and MLAT info and maps the identified boundary 
    points to exact MLAT thresholds based on minimal distance.
    """
    MLAT_all = np.loadtxt(MLAT_file)
    MLT_all = np.loadtxt(MLT_file)
    results = np.zeros((timelength, 2)) * np.nan
    
    for i in range(timelength):
        for j in range(2):
            target_mlt = MLT_all[:, i + IE_OCB_start_index]
            tmp = np.nanargmin(np.abs(boundary[i, j] - target_mlt))
            results[i, j] = MLAT_all[tmp, i + IE_OCB_start_index]
            
    return results

if __name__ == "__main__":
    run_name = 'run25_ideal'
    base_dir = '/Users/weizhang/Desktop/research/EMIC/20181020/'
    path = os.path.join(base_dir, f'{run_name}/IE/')

    if run_name == 'run25_ideal':
        boundary_file_path = os.path.join(path, 'boundary_T32.txt')
        boundary_xlimit = [32, 52]
        GM_OCB_start, GM_OCB_end = 25, 50
        
        MLAT_file = os.path.join(path, 'run25_IE_MLAT_618.txt')
        MLT_file = os.path.join(path, 'run25_IE_MLT_618.txt')
        
    elif run_name == 'run81_epic':
        boundary_file_path = os.path.join(path, 'boundary_T31.txt')
        boundary_xlimit = [31, 49]
        GM_OCB_start, GM_OCB_end = 25, 50
        
        MLAT_file = os.path.join(path, 'run81_IE_MLAT_618.txt')
        MLT_file = os.path.join(path, 'run81_IE_MLT_618.txt')

    # Common Processing
    start_time = boundary_xlimit[0]
    timelength = boundary_xlimit[1] - boundary_xlimit[0] + (1 if run_name == 'run81_epic' else 0)
    
    time4tracing = timelength
    if GM_OCB_end < boundary_xlimit[1]:
        time4tracing = timelength + GM_OCB_end - boundary_xlimit[1]
    
    OCB_MLT = load_boundary(boundary_file_path, start_time, timelength) 
    tmp = load_OCB_all_info(OCB_MLT, MLT_file, MLAT_file, 
                            boundary_xlimit[0] - GM_OCB_start, time4tracing) 
    
    if run_name == 'run25_ideal':
        output_file = os.path.join(path, 'IE_OCB_MLAT_T32.txt')
        np.savetxt(output_file, tmp, fmt='%.3f')
        print(f"Saved matched MLAT info to {output_file}")