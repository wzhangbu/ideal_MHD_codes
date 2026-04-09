#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculations and Data Binning Module.
Handles vector math, coordinate transformations (GSM to SM), and 
spatial data binning into Magnetic Local Time (MLT) grids for Keograms.
"""

import sys
import os
import numpy as np
import glob
import spacepy.pybats.rim as rim
from read_data import Data, get_OCB_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plot
from spacepy import coordinates as coord
from spacepy.time import Ticktock

def GSM2SM(X, time, car=1):
    """
    Converts 3D coordinates from Geocentric Solar Magnetospheric (GSM) 
    to Solar Magnetic (SM) utilizing the Spacepy library.
    """
    # Reshape input vector to N x 3 for Spacepy compatibility
    X_temp = X.reshape(-1, 3)
    X_GSM = coord.Coords(X_temp, 'GSM', 'car')
    
    # Spacepy requires exact timestamps since the Earth's dipole tilts over time
    times = np.repeat(time, len(X_temp[:, 0]), axis=0)
    X_GSM.ticks = Ticktock(times)
    
    # Convert to either Cartesian ('car') or Spherical ('sph') coordinates
    if car == 1:
        X_out = X_GSM.convert('SM', 'car')
    else:
        X_out = X_GSM.convert('SM', 'sph')
    
    # Restore the original array shape
    return X_out.data.reshape(X.shape)

def GetCrossProduct(U, B, method=0):
    """
    Calculates the cross product of Velocity (U) and Magnetic Field (B).
    method=0: Standard U x B (Used for finding Electric Field E = -U x B).
    method=1: U_perp (Extracts velocity strictly perpendicular to the B-field).
    """
    Ux, Uy, Uz = U[..., 0], U[..., 1], U[..., 2]
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
    
    if method == 0:
        # Standard Cross Product Vector Math
        UxB_x = (Uy * Bz - Uz * By)
        UxB_y = (Uz * Bx - Ux * Bz) 
        UxB_z = (Ux * By - Uy * Bx) 
    elif method == 1:
        # U_perp calculation: Subtract the field-aligned velocity from total velocity
        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        udotb = (Ux * Bx + Uy * By + Uz * Bz) / (B_mag**2)
        
        UxB_x = Ux - udotb * Bx
        UxB_y = Uy - udotb * By
        UxB_z = Uz - udotb * Bz  
        
    UxB_mag = np.sqrt(UxB_x**2 + UxB_y**2 + UxB_z**2)
    
    # Reconstruct the 3D array
    UxB = np.zeros((U.shape[0], 3))
    UxB[..., 0], UxB[..., 1], UxB[..., 2] = UxB_x, UxB_y, UxB_z
    
    return UxB, UxB_mag

def get_MLAT(z, Re):
    """Calculates Magnetic Latitude (MLAT) based on Z position and Radial distance."""
    return np.arcsin(z / Re) / np.pi * 180

def getMLT(x, y):
    """Calculates Magnetic Local Time (MLT) based on equatorial X and Y coordinates."""
    mlt = np.arctan2(y, x) / np.pi * 180 / 15 + 12
    return np.mod(mlt, 24)

def read_IE_tracing(file, file_used, time, EMIC_index):
    """
    Reads the Open-Closed Boundary trace limits, converts to MLT, 
    and handles phase-shifting adjustments based on simulation type.
    """
    a = np.loadtxt(file, dtype=float, comments='#')
    boundary_mlt = np.zeros((file_used, 2)) * np.nan

    dawn_SM = a[..., 0:3]
    dusk_SM = a[..., 3:6]
    
    # Extract MLT bounds for Dawn and Dusk sides
    dawn_mlt = getMLT(dawn_SM[..., 0], dawn_SM[..., 1])
    dusk_mlt = getMLT(dusk_SM[..., 0], dusk_SM[..., 1])
    
    boundary_mlt[0:len(dawn_mlt), 0] = dawn_mlt 
    boundary_mlt[0:len(dawn_mlt), 1] = dusk_mlt 

    # Phase shift the arrays depending on the physics model used (Ideal MHD vs EMIC)
    if EMIC_index != 1:
        boundary_mlt = np.roll(boundary_mlt, 3, axis=0)
    else:
        boundary_mlt = np.roll(boundary_mlt, 5, axis=0)

    return boundary_mlt

def get_keogram_data(Data_obj, MLT_range, MLT_OCB, MLAT_OCB, keogram_bin=0.1, SM=0):
    """
    Bins scattered spatial data into discrete MLT grid slices to build a Keogram.
    Extracts the maximum poleward velocity inside each MLT bin near the Open-Closed Boundary.
    """
    # Create uniform MLT bins
    num_bins = int((MLT_range[1] - MLT_range[0]) / keogram_bin)
    mlt_bin = np.linspace(MLT_range[0], MLT_range[1], num_bins + 1)
    
    # Get column indices
    mlt_index = Data_obj.names.index('mlt')
    mlat_index = Data_obj.names.index('mlat')
    uperpx_index = Data_obj.names.index('uperpx')
    
    # If using SM coordinates, compute the Poleward components of Velocity and Force
    if SM == 1:
        iux = Data_obj.names.index('uperpx')
        ix = Data_obj.names.index('x')
        ire = Data_obj.names.index('re')
        
        # 1. Poleward Velocity calculation
        u_hor = -(Data_obj.data[..., iux] * Data_obj.data[..., ix] + 
                  Data_obj.data[..., iux+1] * Data_obj.data[..., ix+1]) / \
                  Data_obj.data[..., ire] / np.cos(np.radians(Data_obj.data[..., mlat_index]))
                  
        u_pole = Data_obj.data[..., iux+2] * np.cos(np.radians(Data_obj.data[..., mlat_index])) + \
                 u_hor * np.sin(np.radians(Data_obj.data[..., mlat_index]))
                 
        if len(np.shape(Data_obj.data)) == 2:
            Data_obj.data = np.concatenate((Data_obj.data, u_pole.reshape(-1, 1)), axis=1)
            Data_obj.names.append('upole')
        upole_index = Data_obj.names.index('upole')
        
        # 2. Poleward Force calculation (JxB + Pressure Gradient)
        ijx = Data_obj.names.index('jx')
        bx_index = Data_obj.names.index('bx')
        jxb, _ = GetCrossProduct(Data_obj.data[..., ijx:ijx+3], Data_obj.data[..., bx_index:bx_index+3], method=0)
        
        igradp = Data_obj.names.index('gradp0')
        gradp = -1 * Data_obj.data[..., igradp:igradp+3] / 6.371
        
        Ftotal = jxb + gradp
        
        F_hor = -(Ftotal[..., 0] * Data_obj.data[..., ix] + Ftotal[..., 1] * Data_obj.data[..., ix+1]) / \
                 Data_obj.data[..., ire] / np.cos(np.radians(Data_obj.data[..., mlat_index]))
                 
        F_pole = Ftotal[..., 2] * np.cos(np.radians(Data_obj.data[..., mlat_index])) + \
                 F_hor * np.sin(np.radians(Data_obj.data[..., mlat_index]))
                 
        if len(np.shape(Data_obj.data)) == 2:
            Data_obj.data = np.concatenate((Data_obj.data, F_pole.reshape(-1, 1)), axis=1)
            Data_obj.names.append('F_pole')
            
    results = np.zeros(len(mlt_bin)) * np.nan
    
    # Loop over each MLT slice to find the maximum values
    for k, mlt_temp in enumerate(mlt_bin):
        # Isolate data within the current MLT slice
        temp = np.argwhere((Data_obj.data[..., mlt_index] >= mlt_temp) & 
                           (Data_obj.data[..., mlt_index] < mlt_temp + keogram_bin))[:, 0] 
                           
        if len(temp) < 1: continue
        data_in_mlt = Data_obj.data[temp, :]
        
        # Find the reference OCB latitude for this MLT
        tmp = np.nanargmin(np.abs(mlt_temp - MLT_OCB))
        mlat_now = MLAT_OCB[tmp]
        
        # Keep only the data physically close to the Open-Closed Boundary (+/- 1 degree)
        lat_tolerance = 2 if getattr(Data_obj, 'sphere_Re', 0) == 2.7 else 1
        lat_mask = np.argwhere((data_in_mlt[..., mlat_index] > mlat_now - lat_tolerance) &
                               (data_in_mlt[..., mlat_index] < mlat_now + lat_tolerance))[:, 0]
                               
        if len(lat_mask) < 1: continue
        data_use = data_in_mlt[lat_mask, :]
        
        # Extract the maximum absolute poleward velocity in this bin
        if SM == 1:
            idx_max = np.nanargmax(np.abs(data_use[..., upole_index]))            
            results[k] = data_use[idx_max, upole_index]
        else:
            idx_max = np.nanargmax(np.abs(data_use[..., uperpx_index]))
            results[k] = data_use[idx_max, uperpx_index]

    return results

def fit_OCB(MLT_IE, MLAT_IE, Re=3, SM=0):
    """
    Fits a quadratic polynomial to the scattered MLAT vs MLT boundary data.
    This creates a smooth, continuous mathematical boundary curve.
    """
    mlat_fit = np.zeros((np.shape(MLAT_IE)[0], np.shape(MLAT_IE)[1]))
    mlt_fit = np.zeros((np.shape(MLT_IE)[0], np.shape(MLT_IE)[1])) if SM == 1 else None

    for i in range(np.shape(MLAT_IE)[1]):
        MLAT_temp = MLAT_IE[:, i]
        MLT_temp = MLT_IE[:, i] if SM == 1 else MLT_IE
        
        valid = ~(np.isnan(MLT_temp) | np.isnan(MLAT_temp))
        if not np.any(valid): continue
        
        # Polyfit returns [A, B, C] for Ax^2 + Bx + C
        coef = np.polyfit(MLT_temp[valid], MLAT_temp[valid], 2)
        mlat_fit[:, i] = np.polyval(coef, MLT_temp)
        
        if SM == 1:
            mlt_fit[:, i] = MLT_temp
            
    return (mlt_fit, mlat_fit) if SM == 1 else (MLT_IE, mlat_fit)

def MLAT2XYZ(MLT, MLAT, Re=9.5):
    """Converts Magnetic Latitude and Local Time back into Cartesian XYZ vectors."""
    num = len(MLT)
    results = np.zeros((num, 3)) * np.nan
    
    for i in range(num):
        angle = (MLT[i] - 12) * 15 * np.pi / 180.
        x = -1.0 if (MLT[i] < 6 or MLT[i] > 18) else 1.0
        y = np.tan(angle) * x
        z = np.sqrt(1 + y**2) * np.tan(np.pi * MLAT[i] / 180.)
        r = np.sqrt(1 + y**2 + z**2)
        
        results[i, 0] = (x / r) * Re
        results[i, 1] = (y / r) * Re
        results[i, 2] = (z / r) * Re
        
    return results