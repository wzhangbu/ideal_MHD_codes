#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading and preprocessing module.
Handles reading CSV files, spatial filtering (dayside/magnetopause), 
2D grid interpolation, and MLT coordinate calculation.
"""

import numpy as np
import os
import sys
from scipy.interpolate import griddata
from spacepy import coordinates as coord
from spacepy.time import Ticktock

# Elementary charge in Coulombs, used for current density calculations
__e = 1.6e-19  

def GSM2SM(X, time, car=1):
    """
    Converts coordinates from Geocentric Solar Magnetospheric (GSM) 
    to Solar Magnetic (SM) using the Spacepy library.
    """
    # Reshape input to an N x 3 array for Spacepy processing
    X_temp = X.reshape(-1, 3)
    X_GSM = coord.Coords(X_temp, 'GSM', 'car')
    
    # Spacepy requires exact timestamps for coordinate rotations 
    # because the Earth's dipole tilt changes with time.
    times = np.repeat(time, len(X_temp[:, 0]), axis=0)
    X_GSM.ticks = Ticktock(times)
    
    # Perform the conversion. Return Cartesian ('car') or Spherical ('sph')
    if car == 1:
        X_out = X_GSM.convert('SM', 'car')
    else:
        X_out = X_GSM.convert('SM', 'sph')
    
    # Restore the original array shape before returning
    return X_out.data.reshape(X.shape)

class Data:
    """Wrapper class for managing and transforming simulation data."""
    X_LINE = None
    __e = 1.6e-19
    
    def __init__(self, data=None, names=None, X_LINE=0, SM=0):
        self.data = data
        self.names = names
        self.SM = SM
        self.X_LINE = X_LINE
                             
    @staticmethod
    def _read_from_file_ideal(data_obj, path, axis1range, axis2range, xline=1, SM=0, file=None):
        """Reads ideal MHD CSV data, renames axes, and applies spatial filtering."""
        # Load CSV data into a structured NumPy array
        data = np.genfromtxt(path, dtype=float, delimiter=",", names=True)
        
        # Standardize column names: lowercase, remove underscores
        names = [name.lower().replace("_", "") for name in data.dtype.names]
        # Rename ParaView's default point coordinate names to standard x, y, z
        names = [name.replace("points0", "x").replace("points1", "y").replace("points2", "z") for name in names]
        
        # Convert structured array to standard 2D float array for easier math operations
        data_array = np.array([data[name] for name in data.dtype.names]).T
        data_obj.data = data_array
        data_obj.names = names
        data_obj.SM = SM
        data_obj.X_LINE = 1 if xline == 1 else 0
            
        # Filter out nightside/unwanted regions to save memory and processing time
        data_obj._Data__get_dayside(axis1range, axis2range)
        
        # Coordinate Transformations to Solar Magnetic (SM)
        if SM == 0:
            if file is None:
                sys.exit('Time error: Please provide a valid file name to extract time.')
            # Extract time string from filename (e.g., 't30.csv' -> '2018-10-20T21:30')
            time = '2018-10-20T21:' + file[-6:-4]
            ix = names.index('x')
            
            # Extract GSM coordinates and convert in-place
            X_GSM = data_obj.data[..., ix:ix + 3]
            data_obj.data[..., ix:ix + 3] = GSM2SM(X_GSM, time, car=1)
            
        elif SM == 1:
            # If SM=1, rotate specific vector fields (B, J, U, gradP, etc.) into the SM frame
            convert_index = [names.index('bx'), names.index('gradbx0'), names.index('gradby0'), 
                             names.index('gradbz0'), names.index('gradp0'), names.index('gradpb0'),
                             names.index('jx'), names.index('x'), names.index('normals0'), names.index('ux')]
            
            time = '2018-10-20T21:' + file[-6:-4]
            for i in convert_index:
                X_GSM = data_obj.data[..., i:i + 3]
                data_obj.data[..., i:i + 3] = GSM2SM(X_GSM, time, car=1)
        
        # Calculate MLT for the loaded coordinates
        data_obj._get_MLT()
        return data_obj
    
    @staticmethod
    def _read_from_file_EMIC(data_obj, path, axis1range, axis2range, s=25, scaling=16, xline=1):
        """Reads EMIC (multi-fluid/PIC) CSV data, calculates currents, and applies spatial filtering."""
        data = np.genfromtxt(path, dtype=float, delimiter=",", names=True)
        names = [name.lower().replace("_", "") for name in data.dtype.names]
        names = [name.replace("points0", "x1").replace("points1", "y1").replace("points2", "z1") for name in names]
        
        data_array = np.array([data[name] for name in data.dtype.names]).T

        # Calculate total Current Density (J) from multi-fluid velocities:
        # Formula: J = e * (n_i * V_i - n_e * V_e)
        J = np.zeros((len(data_array[:, 0]), 3))
        rho_i = data_array[:, names.index('rhos1')]
        rho_e = data_array[:, names.index('rhos0')]
        Ue = data_array[:, names.index('uxs0'):names.index('uxs0') + 3]
        Ui = data_array[:, names.index('uxs1'):names.index('uxs1') + 3]
        
        for i in range(3):
            # Applying scaling factors specific to the EMIC simulation setup
            J[:, i] = (rho_i * Ui[:, i] - rho_e * Ue[:, i] * s) * 1e15 * Data.__e / scaling
            
        names.extend(['jx', 'jy', 'jz'])
        data_array = np.concatenate((data_array, J), axis=1)
        
        data_obj.data = data_array
        data_obj.names = names
        data_obj.X_LINE = 1 if xline == 1 else 0
        data_obj._Data__get_dayside(axis1range, axis2range)
        
        return data_obj
        
    def __get_dayside(self, axis1range, axis2range):
        """Filters out nightside points using a parabolic magnetopause boundary approximation."""
        x_idx, y_idx, z_idx = self.names.index('x'), self.names.index('y'), self.names.index('z')
        X, Y, Z = self.data[:, x_idx], self.data[:, y_idx], self.data[:, z_idx]
        
        # Spatial filtering equation for dayside bounding (similar to Shue et al. empirical models).
        # We only keep points where X is greater than the parabolic boundary shape.
        valid_idx = np.argwhere(X > (10 - 0.05 * (Y + 0.5)**2 - 0.06 * Z**2))[:, 0]
        results = self.data[valid_idx, :]
        
        if self.X_LINE != 1:
            # Filter strictly within the Z-axis limits provided
            Z1 = results[:, z_idx]
            z_valid = np.argwhere((Z1 >= axis2range[0]) & (Z1 < axis2range[1]))[:, 0]                
            results = results[z_valid, :]
            
            # Sub-grid variance filtering to remove overlapping/anomalous boundary noise
            Y1, X1, Z1 = results[:, y_idx], results[:, x_idx], results[:, z_idx]
            step = 0.1 if axis2range[0] * axis2range[1] < 0 else 0.15
            Zbin = np.arange(axis2range[0], axis2range[1] + step, step)
            Ybin = np.arange(axis1range[0], axis1range[1] + step, step)
            
            rm_mark = []
            # Scan through the grid space using small steps
            for ytemp in Ybin:     
                ztemp = 0 if axis2range[0] * axis2range[1] < 0 else axis2range[0] + step
                temp = np.argwhere((Z1 >= ztemp) & (Z1 < ztemp + step))[:, 0]
                temp1 = np.argwhere((Y1[temp] > ytemp) & (Y1[temp] < ytemp + step))[:, 0]
                var_std = np.var(X1[temp][temp1]) if len(X1[temp][temp1]) > 0 else 0
                
                # If variance in X is unusually high, flag points below the mean for removal
                for ztemp_sub in Zbin:
                    temp_sub = np.argwhere((Z1 >= ztemp_sub) & (Z1 < ztemp_sub + step))[:, 0]
                    temp1_sub = np.argwhere((Y1[temp_sub] > ytemp) & (Y1[temp_sub] < ytemp + step))[:, 0]
                    
                    var = np.var(X1[temp_sub][temp1_sub]) if len(X1[temp_sub][temp1_sub]) > 0 else 0
                    if var > 5 * var_std and var_std > 0:
                        mean_x = np.mean(X1[temp_sub][temp1_sub])
                        temp2 = np.argwhere(X1[temp_sub][temp1_sub] < mean_x)[:, 0]
                        rm_mark.append(temp_sub[temp1_sub][temp2])
                        
            # Apply the removal mask
            if len(rm_mark) > 1:
                rm_mark1 = np.concatenate(rm_mark)
                results[rm_mark1, :] = np.nan
                # Rebuild array dropping any row with a NaN in the 'X' coordinate
                results = results[~np.isnan(results[:, x_idx]), :]
            
        self.data = results
        
    def _convert_to_2d(self, coord_name1="x", coord_name2="z", nc1=101, nc2=21, c1range=None, c2range=None):  
        """Interpolates unstructured scatter data onto a structured 2D uniform grid."""
        coord1_index = self.names.index(coord_name1)
        coord2_index = self.names.index(coord_name2)
        coord1 = self.data[:, coord2_index]
        coord2 = self.data[:, coord1_index]

        c1range = c1range or [min(coord1), max(coord1)]
        c2range = c2range or [min(coord2), max(coord2)]

        # Generate target uniform mesh grids
        c1_grid = np.linspace(c1range[0], c1range[1], nc1)
        c2_grid = np.linspace(c2range[0], c2range[1], nc2)
        c1_grid, c2_grid = np.meshgrid(c2_grid, c1_grid)
        
        # Nearest neighbor interpolation projects the scattered 3D slice onto the uniform 2D grid
        data_2d = griddata((coord1, coord2), self.data, (c1_grid, c2_grid), method="nearest")
        return Data(data_2d, self.names)

    def _smooth_data(self, n):
        """Applies a simple 5-point moving average smoothing filter 'n' times."""
        # Iteratively smooth data to reduce high-frequency simulation noise
        for _ in range(n):
            new_data = self.data.copy()
            for i in range(1, self.data.shape[0] - 1):
                for j in range(1, self.data.shape[1] - 1):
                    # Center pixel contributes 50%, neighbors contribute 12.5% each
                    self.data[i, j] = 0.5 * new_data[i, j] + 0.125 * (
                        new_data[i - 1, j] + new_data[i + 1, j] + 
                        new_data[i, j - 1] + new_data[i, j + 1]
                    )
            
    def _get_MLT(self):
        """Calculates Magnetic Local Time (MLT) based on equatorial X and Y coordinates."""
        x_idx, y_idx = self.names.index('x'), self.names.index('y')
        x, y = self.data[..., x_idx], self.data[..., y_idx]
        
        # atan2 gives angle in radians from X-axis. 
        # Divide by pi, scale to 180 deg, divide by 15 deg/hour, and shift by 12 hours.
        mlt = np.arctan2(y, x) / np.pi * 180 / 15 + 12
        # Ensure values wrap cleanly between 0 and 24 hours
        mlt = np.mod(mlt, 24)
        
        # Append MLT as a new feature column to the underlying dataset
        if len(np.shape(self.data)) == 3:
            mlt = mlt.reshape(self.data.shape[0], self.data.shape[1], 1)
            self.data = np.concatenate((self.data, mlt), axis=2)
        elif len(np.shape(self.data)) == 2:
            mlt = mlt.reshape(-1, 1)
            self.data = np.concatenate((self.data, mlt), axis=1)
            
        if 'mlt' not in self.names:
            self.names.append('mlt')