#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Driver Script.
Iterates through CSV datasets, filters data, computes the T-Plot physical variables 
across the magnetopause boundaries, constructs the Keogram MLT grids, and outputs final visualizations.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from read_data import Data
import calculation
import plot
from plot import PlotKeogram

# --- Global Toggles & Settings ---
X_LINE = 0         # 1: Extract and focus exclusively on Reconnection X-line limits. 0: Process global domain.
EMIC_index = 0     # 1: Use EMIC (Multi-fluid/PIC) simulation processing. 0: Use Ideal MHD.
SAVE = 1           # 1: Generate and save plot images. 0: Dry run.
SM = 1             # 1: Transform coordinate frames to Solar Magnetic (SM). 0: Keep in GSM.
VNORM = 1          # 1: Use visual limits for colormaps. 0: Autoscale.
TANGENTIAL = 0     # 1: Compute tangential force/velocity components across the boundary.

def Keogram_MLT(Data_obj, tplotvar, mltrange, mltTick, method='sign', By=-10):
    """
    Bins 2D spatial parameter data into an MLT grid for a single time step.
    This collapses a full spatial volume into a 1D array by extracting the
    maximum or mean value inside each MLT slice.
    """
    iMLT = Data_obj.names.index('mlt')
    mlt = Data_obj.data[..., iMLT]
    
    # Calculate the number of MLT bins based on the requested range and tick resolution
    y_number = int((mltrange[1] - mltrange[0]) / mltTick + 1)
    
    if len(np.shape(tplotvar)) < 3:
        # Processing a single 2D variable layer
        results = np.zeros(y_number)
        for i in range(y_number):
            mlt_temp = mltrange[0] + i * mltTick
            temp = np.argwhere(abs(mlt - mlt_temp) < mltTick / 2)
            if len(Data_obj.data[temp[:, 0], temp[:, 1], 0]) < 1:
                results[i] = np.nan
                continue
            # Extract Max or Mean depending on the argument
            results[i] = np.nanmean(tplotvar[temp[:, 0], temp[:, 1]]) if method == 'mean' else np.nanmax(tplotvar[temp[:, 0], temp[:, 1]])
            
    elif len(np.shape(tplotvar)) == 3:
        # Processing multiple variables stacked in a 3D array [Y, Z, Var]
        results = np.zeros((y_number, np.shape(tplotvar)[2]))
            
        for i in range(y_number):
            mlt_temp = mltrange[0] + i * mltTick
            temp = np.argwhere(abs(mlt - mlt_temp) < mltTick / 2)
            
            # Skip if no data falls into this MLT bin
            if len(mlt[temp[:, 0], temp[:, 1]]) < 1:
                results[i, :] = np.nan
                continue
                
            for j in range(tplotvar.shape[2]):
                if method != 'sign':
                    results[i, j] = np.nanmax(tplotvar[temp[:, 0], temp[:, 1], j])
                else:
                    # Sign-sensitive extraction: Take the value that has the maximum absolute magnitude
                    if np.isnan(tplotvar[temp[:, 0], temp[:, 1], j]).all():
                        results[i, j] = np.nan
                        continue
                    temp1 = np.nanargmax(np.abs(tplotvar[temp[:, 0], temp[:, 1], j]))
                    results[i, j] = tplotvar[temp[temp1, 0], temp[temp1, 1], j]
                    
    return results


if __name__ == "__main__":
    
    # 1. Define interpolation grid limits (Y and Z bounds in Earth Radii)
    axis1range = [-15, 15.]
    axis2range = [-10, 10] 
    axis1num = 601
    axis2num = int((axis2range[1] - axis2range[0]) * 20 + 1)
    
    # Restrict volume if we only care about the X-Line
    if X_LINE != 1:
        axis2range = [7, 8.5]
        axis2num = int((axis2range[1] - axis2range[0]) * 15 + 1)
     
    path = '/Users/weizhang/Desktop/research/EMIC/20181020/run25_ideal/MP_Bz-5/'
    
    # 2. Define Variables to compute
    tplotnames = ['bx', 'by', 'bz', 'ux', 'uy', 'uz', 
                  'gradp0', 'gradp1', 'gradp2', 'gradpb0', 'gradpb1', 'gradpb2', 
                  'uperpx', 'uperpy', 'uperpz', 
                  'Ex', 'Ey', 'Ez', 'JxBx', 'JxBy', 'JxBz', 'Tensionx', 'Tensiony', 'Tensionz', 
                  'Ftotalx', 'Ftotaly', 'Ftotalz', 'JxBcalx', 'JxBcaly', 'JxBcalz',
                  'Ftotalcalx', 'Ftotalcaly', 'Ftotalcalz', 'upole']

    # The subset of variables to plot natively on the Keogram array
    tplot_GEM = ['ux', 'uz', 'upole', 'JxBy', 'gradp1', 'Ftotaly', 
                 'Tensiony', 'gradpb1', 'JxBy', 'Ftotalx', 'Ftotaly', 'Ftotalz',
                 'uperpx', 'uperpy', 'uperpz']  
    
    keogram_path = '/Users/weizhang/Desktop/research/EMIC/20181020/run25_ideal/UpperMP/Bz_-5/'
    fig_path = keogram_path + 'snapshots/Z = ' + str(axis2range[0]) + '/'
    IE2MP = 'IE2MP_Bnorm.txt'
    Tracing_path = '/Users/weizhang/Desktop/research/EMIC/20181020/run25_ideal/'
    
    # 3. Path Management
    keogram_path_GEM = keogram_path + 'GEM/'
    if SM == 0:
        keogram_path_GEM += 'GSM/'
        fig_path += 'GSM/'
    elif SM == 1:
        keogram_path_GEM += 'SM/'
        fig_path += 'SM/'
        
    if SAVE != 0:
        os.makedirs(fig_path, exist_ok=True)
        os.makedirs(keogram_path_GEM, exist_ok=True)
    else:
        fig_path, keogram_path, keogram_path_GEM = None, None, None
    
    os.chdir(path)
    files = sorted(glob.glob('./t*.csv'))
    
    mltRange = [6, 18]
    mltTick = 0.1
    # Initialize the 3D Master Array: [MLT_BINS, TIME_STEPS, VARIABLES]
    TplotVarKeogram = np.zeros((int((mltRange[1] - mltRange[0]) / mltTick + 1), len(files), len(tplotnames)))
    
    # 4. Master Time Loop
    for k, file in enumerate(files):
        print(f"Processing: {file}")
        
        # Step A: Load CSV Data
        data1d = Data()
        if EMIC_index == 0:
            data1d = Data._read_from_file_ideal(data1d, os.path.join(path, file), axis1range, axis2range, xline=X_LINE, SM=SM, file=file)    
        elif EMIC_index == 1:
            data1d = Data._read_from_file_EMIC(data1d, os.path.join(path, file), axis1range, axis2range, xline=X_LINE)          
        
        # Step B: Interpolate from unstructured scatter to uniform 2D plane
        data2d = data1d._convert_to_2d("y", "z", axis1num, axis2num, axis1range, axis2range)
  
        xline = None
        if X_LINE == 1:
            data2d, xline = calculation.getXlineBnorm(data2d, zband=40, BnormUezDiff=20, EMIC_index=EMIC_index)
                
        data = data2d.data
        names = data2d.names

        # Step C: Compute Physics arrays (Forces, Cross Products)
        if EMIC_index == 0:
            TplotVar = calculation.GetTplotNames_ideal_tang(data2d, tplotnames, SM, xline, file, TANGENTIAL)
        elif EMIC_index == 1:
            TplotVar = calculation.GetTplotNames_EMIC(data2d, tplotnames, xline)

        # Step D: Bin the calculated arrays into the MLT Grid
        TplotVarKeogram[:, k, :] = Keogram_MLT(data2d, TplotVar, mltRange, mltTick, method='sign')

    # Re-wrap the Keogram array back into the Data class for PlotKeogram compatibility
    tplot_para = Data(TplotVarKeogram, tplotnames, X_LINE=X_LINE, SM=SM)
    IE_tracing = calculation.read_IE_tracing(os.path.join(Tracing_path, IE2MP), len(files), '2018-10-20T21:40', EMIC_index, X_LINE)
    
    # 5. Plotting Execution
    keogram_plot = PlotKeogram(tplot_para, mltRange, mltTick, len(files), boundary=0)
    keogram_plot._plot_keogram_MLT_GEM(tplot_GEM, MLT_limit=[6, 18], vnorm=VNORM,
                                       save_path=keogram_path_GEM, 
                                       IE_tracing=IE_tracing, EMIC_index=EMIC_index, 
                                       small_zlim=0, plot_num=3)