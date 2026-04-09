#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for reading data, computing coordinates (MLT, MLAT), 
and handling coordinate transformations (GSM to SM).
"""

import os
import numpy as np
from spacepy import coordinates as coord
from spacepy.time import Ticktock

# Elementary charge constant
E_CHARGE = 1.6e-19

def get_OCB_data(path):
    """Reads OCB data from a CSV file."""
    return np.genfromtxt(path, dtype=float, delimiter=",")

def GSM2SM(X, time, car=1):
    """
    Converts coordinates from Geocentric Solar Magnetospheric (GSM) 
    to Solar Magnetic (SM).
    """
    X_GSM = coord.Coords(X, 'GSM', 'car')
    times = np.repeat(time, len(X[:, 0]), axis=0)
    X_GSM.ticks = Ticktock(times)
    
    if car == 1:
        X_out = X_GSM.convert('SM', 'car')
    else:
        X_out = X_GSM.convert('SM', 'sph')
    
    return X_out.data

class Data:
    """Class to store and process magnetospheric simulation data."""
    
    def __init__(self, data=None, names=None):
        self.data = data
        self.names = names
        self.SM = 0
        self.RE = 9.5
                 
    @classmethod
    def read_from_file_ideal(cls, path, SM=0):
        """Reads ideal MHD data from a CSV file and initializes a Data object."""
        data_raw = np.genfromtxt(path, dtype=float, delimiter=",", names=True)
        
        # Format column names to lowercase and standard axes
        names = [name.lower().replace("_", "") for name in data_raw.dtype.names]
        names = [name.replace("points0", "x").replace("points1", "y").replace("points2", "z") for name in names]
        
        data_array = np.array([data_raw[name] for name in data_raw.dtype.names]).T
        
        obj = cls(data=data_array, names=names)
        
        if SM != 0:
            ix = names.index('x')
            time = f'2018-10-20T21:{path[-6:-4]}'
            
            # Convert Position
            X_GSM = obj.data[..., ix:ix+3]
            obj.data[..., ix:ix+3] = GSM2SM(X_GSM, time, car=1)
            
            # Convert Velocity (if uperpx exists)
            if 'uperpx' in names:
                iux = names.index('uperpx')
                Ux_GSM = obj.data[..., iux:iux+3]
                obj.data[..., iux:iux+3] = GSM2SM(Ux_GSM, time, car=1)
                
        obj.get_MLT()
        obj.get_MLAT()    
        return obj
    
    def get_MLT(self):
        """Calculates Magnetic Local Time (MLT) based on X and Y coordinates."""
        x_index, y_index = self.names.index('x'), self.names.index('y')
        x, y = self.data[..., x_index], self.data[..., y_index]
        
        mlt = np.arctan2(y, x) / np.pi * 180 / 15 + 12
        mlt = np.mod(mlt, 24)
        
        if len(np.shape(self.data)) == 3:
            mlt = mlt.reshape(self.data.shape[0], self.data.shape[1], 1)
            self.data = np.concatenate((self.data, mlt), axis=2)
        elif len(np.shape(self.data)) == 2:
            mlt = mlt.reshape(-1, 1)
            self.data = np.concatenate((self.data, mlt), axis=1)
            
        self.names.append('mlt')

    def get_MLAT(self):
        """Calculates Magnetic Latitude (MLAT) based on Z and radial distance."""
        z_index = self.names.index('z')
        z = self.data[..., z_index]
        
        re_index = self.names.index('re')
        self.RE = self.data[..., re_index]
        
        mlat = np.arcsin(z / self.RE) / np.pi * 180
        
        if len(np.shape(self.data)) == 3:
            mlat = mlat.reshape(self.data.shape[0], self.data.shape[1], 1)
            self.data = np.concatenate((self.data, mlat), axis=2)
        elif len(np.shape(self.data)) == 2:
            mlat = mlat.reshape(-1, 1)
            self.data = np.concatenate((self.data, mlat), axis=1)
            
        self.names.append('mlat')

if __name__ == "__main__":
    # Test execution
    sphere_Re = 4
    path = f'/Users/weizhang/Desktop/research/EMIC/20181020/run25_ideal/RE_Uperp/{sphere_Re}RE/'
    os.chdir(path)
    fs = './t40.csv'
    
    results = Data.read_from_file_ideal(path + fs)
    ix, iy, iz = results.names.index("x"), results.names.index("y"), results.names.index("z")
    
    avg_x = np.sqrt((results.data[0:-1, ix] - results.data[1:, ix]) ** 2 +
                    (results.data[0:-1, iy] - results.data[1:, iy]) ** 2 + 
                    (results.data[0:-1, iz] - results.data[1:, iz]) ** 2)
    print(f'Average distance between 2 adjacent points is {np.mean(avg_x):.4f}')