#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main driver script.
Iterates through CSV datasets, filters data, computes the T-Plot physical variables 
across the magnetopause boundaries, constructs the Keogram MLT grids, and outputs final visualizations.
"""

import os
import glob
import numpy as np
from read_data import Data
from plot import PlotKeogram
import calculation

# --- Global Toggles & Settings ---
X_LINE = 1         # 1: Extract and focus exclusively on Reconnection X-line limits. 0: Process all data.
EMIC_index = 0     # 1: Use EMIC (Multi-fluid/PIC) simulation processing. 0: Use Ideal MHD.
SAVE = 1           # 1: Generate and save plot images. 0: Run data generation only.
SM = 1             # 1: Transform coordinate frames to Solar Magnetic (SM). 0: Keep in GSM.
VNORM = 1          # 1: Use hardcoded symmetric visual limits for colormaps. 0: Autoscale to data max/min.
TANGENTIAL = 1     # 1: Compute tangential force/velocity components across the boundary.

def Keogram_MLT(Data_obj, tplotvar, mltrange, mltTick, method='sign', By=-10):
    """
    Bins 2D spatial parameter data into an MLT grid for a single time step.
    This collapses a full spatial volume into a 1D line by extracting either the
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
            
            # Skip if no data falls into this MLT bin
            if len(Data_obj.data[temp[:, 0], temp[:, 1], 0]) < 1:
                results[i] = np.nan
                continue
            
            # Condense the slice using mean or max operations
            results[i] = np.nanmean(tplotvar[temp[:, 0], temp[:, 1]]) if method == 'mean' else np.nanmax(tplotvar[temp[:, 0], temp[:, 1]])
    else:
        # Processing multiple variables stacked in a 3D array [Y, Z, Var]
        results = np.zeros((y_number, np.shape(tplotvar)[2]))
        for i in range(y_number):
            mlt_temp = mltrange[0] + i * mltTick
            temp = np.argwhere(abs(mlt - mlt_temp) < mltTick / 2)
            
            if len(mlt[temp[:, 0], temp[:, 1]]) < 1:
                results[i, :] = np.nan
                continue
                
            for j in range(tplotvar.shape[2]):
                if method != 'sign':
                    # Standard extraction: just take the maximum value in the bin
                    results[i, j] = np.nanmax(tplotvar[temp[:, 0], temp[:, 1], j])
                else:
                    # Sign-sensitive extraction: Take the value that has the maximum absolute magnitude
                    if np.nanmax(By) == -10:
                        if np.isnan(tplotvar[temp[:, 0], temp[:, 1], j]).all():
                            results[i, j] = np.nan
                            continue
                        temp1 = np.nanargmax(np.abs(tplotvar[temp[:, 0], temp[:, 1], j]))
                        results[i, j] = tplotvar[temp[temp1, 0], temp[temp1, 1], j]
                    else:
                        # Alternatively, extract the parameter specifically where By is maximized
                        Uz1 = tplotvar[temp[:, 0], temp[:, 1], j]
                        By1 = By[temp[:, 0], temp[:, 1]]
                        temp1 = np.nanargmax(np.abs(By1))
                        results[i, j] = Uz1[temp1[:, 0], temp1[:, 1], j]
                        
    return results

if __name__ == "__main__":
    
    # Define interpolation grid limits (Y and Z bounds in Earth Radii)
    axis1range = [-15, 15.]
    axis2range = [-10, 10]
    axis1num = 601
    axis2num = int((axis2range[1] - axis2range[0]) * 20 + 1)
    
    # Adjust boundaries if not running in X-line specific mode
    if X_LINE != 1:
        axis2range = [7, 8.5]
        axis2num = int((axis2range[1] - axis2range[0]) * 15 + 1)
        
    path = '/Users/weizhang/Desktop/research/EMIC/20181020/run25_ideal/x_line_force/'
    keogram_path = '/Users/weizhang/Desktop/research/EMIC/20181020/run25_ideal/UpperMP/'
    
    # The default variables to process and plot
    tplotnames = [
        'bx', 'by', 'bz', 'ux', 'uy', 'uz', 'gradp0', 'gradp1', 'gradp2', 'gradpb0', 'gradpb1', 'gradpb2', 
        'uperpx', 'uperpy', 'uperpz', 'Ex', 'Ey', 'Ez', 'JxBx', 'JxBy', 'JxBz', 'Tensionx', 'Tensiony', 'Tensionz', 
        'Ftotalx', 'Ftotaly', 'Ftotalz', 'JxBcalx', 'JxBcaly', 'JxBcalz', 'Ftotalcalx', 'Ftotalcaly', 'Ftotalcalz', 'upole'
    ]
    tplot_GEM = ['uperpx', 'uz', 'upole', 'JxBx', 'gradp0', 'Ftotalx', 'Tensionx', 'gradpb0', 'JxBx']

    # Append Tangential parameters if the toggle is set
    if TANGENTIAL == 1:
        SM = 0
        keogram_path += 'tangential/'
        tplotnames.extend(['JxB_pole', 'gradp_pole', 'Ftotal_pole', 'Ftention_pole', 'Gradpb_pole', 'Ftotal_tangx',
                           'Ftotal_tangy', 'Ftotal_tangz', 'U_perp_tangx', 'U_perp_tangy', 'U_perp_tangz', 'uy_tangy', 'uy_tangy+Va', 'Va'])
        tplot_GEM = ['uperpy', 'uy', 'upole', 'JxB_pole', 'gradp_pole', 'Ftotal_pole', 'Ftention_pole', 'Gradpb_pole', 'JxB_pole',
                     'Ftotal_tangx', 'Ftotal_tangy', 'Ftotal_tangz', 'U_perp_tangx', 'U_perp_tangy', 'U_perp_tangz', 'uy_tangy', 'uy_tangy+Va', 'Va']

    fig_path = keogram_path + 'snapshots/' + 'Z = ' + str(axis2range[0]) + '/'

    # Reroute save paths if dealing strictly with X-line data
    if X_LINE == 1:
        keogram_path = '/Users/weizhang/Desktop/research/EMIC/20181020/run25_ideal/UpperMP/xline/'
        if TANGENTIAL == 1:
            SM = 0
            keogram_path = '/Users/weizhang/Desktop/research/EMIC/20181020/run25_ideal/UpperMP/tangential/xline/'
        fig_path = keogram_path + 'snapshots/'

    IE2MP = 'IE2MP_Bnorm.txt'
    Tracing_path = '/Users/weizhang/Desktop/research/EMIC/20181020/run25_ideal/'
    
    # Establish sub-directories for frame-of-reference
    keogram_path_GEM = keogram_path + 'GEM/'
    keogram_path_GEM += 'SM/' if SM == 1 else 'GSM/'
    fig_path += 'SM/' if SM == 1 else 'GSM/'
    
    # Make directories if writing files to disk is active
    if SAVE == 1:
        os.makedirs(fig_path, exist_ok=True)
        os.makedirs(keogram_path_GEM, exist_ok=True)
    else:
        fig_path, keogram_path, keogram_path_GEM = None, None, None
    
    os.chdir(path)
    files = sorted(glob.glob('./t*.csv'))
    
    # MLT Keogram Definitions
    mltRange = [6, 18]
    mltTick = 0.1
    # Initialize the 3D Master Array: [MLT_BINS, TIME_STEPS, VARIABLES]
    TplotVarKeogram = np.zeros((int((mltRange[1] - mltRange[0]) / mltTick + 1), len(files), len(tplotnames)))
    
    # Loop over the CSV data sequence (subset to [15:16] for rapid testing, adjust for full run)
    for k, file in enumerate(files[15:16]): 
        print(f"Processing: {file}")
        
        # 1. Load Data
        data1d = Data()
        if EMIC_index == 0:
            data1d = Data._read_from_file_ideal(data1d, path + file, axis1range, axis2range, xline=X_LINE, SM=SM, file=file)    
        else:
            data1d = Data._read_from_file_EMIC(data1d, path + file, axis1range, axis2range, xline=X_LINE)          
        
        # 2. Interpolate from unstructured scatter to uniform 2D plane
        data2d = data1d._convert_to_2d("y", "z", axis1num, axis2num, axis1range, axis2range)
        
        # 3. Detect X-Line (if toggled)
        xline = None
        if X_LINE == 1:
            uz_index = data2d.names.index('uz')
            # Extract points where vertical flow velocity is near stagnant
            temp = np.abs(data2d.data[..., uz_index]) < 1.5
            xline = data2d.data.copy()
            xline[~temp, :] = np.nan   
                
        # 4. Calculate Physics Variables (Cross products, Tensions, Gradients)
        if EMIC_index == 0:
            TplotVar = calculation.GetTplotNames_ideal_tang(data2d, tplotnames, SM, xline, file, TANGENTIAL) if TANGENTIAL else calculation.GetTplotNames_ideal(data2d, tplotnames, SM, xline, file)

        # 5. Bin into the MLT Keogram Array
        if X_LINE == 0:
            TplotVarKeogram[:, k, :] = Keogram_MLT(data2d, TplotVar, mltRange, mltTick, method='sign')
        elif X_LINE == 1:
            TplotVarKeogram[:, k, :] = Keogram_MLT(data2d, TplotVar, mltRange, mltTick, method='max', By=TplotVar[..., 33])
            
    # Wrap the Keogram array back into the Data class for plotting compatibility
    tplot_para = Data(TplotVarKeogram, tplotnames, X_LINE=X_LINE, SM=SM)
    IE_tracing = calculation.read_IE_tracing(Tracing_path + IE2MP, len(files), '2018-10-20T21:40', EMIC_index, X_LINE)
    
    # Calculate specialized aggregate variable if requested
    if 'uy_tangy+Va' in tplotnames:
        tplot_para.data[..., -2] = tplot_para.data[..., -1] + abs(tplot_para.data[..., -3])
        
    # Execute the final Matplotlib routines
    keogram_plot = PlotKeogram(tplot_para, mltRange, mltTick, len(files), boundary=0)
    keogram_plot._plot_keogram_MLT_GEM(tplot_GEM, MLT_limit=[6, 18], vnorm=VNORM, save_path=keogram_path_GEM, IE_tracing=IE_tracing, EMIC_index=EMIC_index, small_zlim=0, plot_num=3)