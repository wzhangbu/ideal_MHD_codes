#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keogram & Convection Mapping Visualizer.
Uses spacepy.pybats to plot ionospheric convection and tracks 
phenomena expansion speeds via linear regression.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import spacepy.pybats.rim as rim
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import calculation
from calculation import KeogramData

def convection(file_path, run_name, start_index, file_num, IE_OCB=None, save_path=None, save_name=None):
    """
    Plots 2D Ionospheric Convection maps using PyBats RIM outputs (.idl files).
    Overlays quiver plots representing ionospheric flows.
    """
    print('Starting to plot the convection map...')
    
    if IE_OCB is not None:
        IE_start, IE_end = 2125, 2150
        IE_MLT_file = f"{run_name[0:5]}_IE_MLT_618.txt"
        IE_MLAT_file = f"{run_name[0:5]}_IE_MLAT_618.txt"
        MLT_IE_all = calculation.get_IE_OCB_data(os.path.join(file_path, IE_MLT_file))
        MLAT_IE_all = calculation.get_IE_OCB_data(os.path.join(file_path, IE_MLAT_file))
    
    os.chdir(file_path)  
    files = sorted(glob.glob('./it*.idl'))
    print(f'There are {len(files)} IDL files in total.')
    
    for k, filename in enumerate(files):
        if k < start_index or k > (start_index + file_num):
            continue

        print(f"Plotting: {filename}")
        fig = plt.figure(figsize=(8, 12))
        ie = rim.Iono(filename)
        
        # Calculate 2D horizontal flows and poleward velocity
        ie["n_r"] = np.sqrt(ie["n_x"]**2 + ie["n_y"]**2)
        ie["V_pole"] = -(ie["n_ux"] * ie["n_x"] + ie["n_uy"] * ie["n_y"]) / ie["n_r"]        
        ie["n_u"] = np.sqrt(ie["n_ux"]**2 + ie["n_uy"]**2 + ie["n_uz"]**2) 
        
        vars_to_plot = ["V_pole", "n_ux"]
        latrange = 18

        time = int(ie.meta['file'].split('/')[-1][9:13])
        if IE_OCB is not None:
            if time < IE_start:
                MLT_IE, MLAT_IE = MLT_IE_all[:, 0], MLAT_IE_all[:, 0]
            elif time > IE_end:
                MLT_IE, MLAT_IE = MLT_IE_all[:, -1], MLAT_IE_all[:, -1]
            else:
                MLT_IE, MLAT_IE = MLT_IE_all[:, time - IE_start], MLAT_IE_all[:, time - IE_start]
            
            valid = ~(np.isnan(MLT_IE) | np.isnan(MLAT_IE))
            mlt_origin, mcol_origin = MLT_IE[valid], 90. - MLAT_IE[valid]
            mlong_origin = ((mlt_origin - 12) * 15 + 90) / 180. * np.pi
            
            MLT_IE_prime, MLAT_IE_prime = calculation.fit_OCB(MLT_IE, MLAT_IE)           
            mcol = 90. - MLAT_IE_prime
            mlong = ((MLT_IE_prime - 12) * 15 + 90) / 180. * np.pi
        
        # Generate contour plots for each variable
        for i, var in enumerate(vars_to_plot):
            loc = 210 + 1 + i
            if var == "V_pole": maxz, cmap = 0.6, 'viridis'
            elif var == "n_u": maxz, cmap = 0.65, 'jet'
            elif var == 'n_ux': maxz, cmap = 0.9, 'bwr'
            else: maxz, cmap = 15, 'jet'
                
            fig, ax, cnt, cb = ie.add_cont(var, add_cbar=True, target=fig, loc=loc, lines=False, minz=0, maxz=maxz, cmap=cmap, extend='both')
            
            # Format out-of-bounds colors
            if cmap == 'viridis':
                cnt.cmap.set_under(color='black')
                cnt.cmap.set_over(color='yellow')
            else:
                cnt.cmap.set_under(color='blue')
                cnt.cmap.set_over(color='darkred')
                
            cb.set_label('m/s')
            ax.set_ylim(0, latrange)
            
            # Overlay Boundary Scatter Points
            if IE_OCB is not None:
                ax.scatter(mlong_origin, mcol_origin, [15]*len(mlt_origin), color='magenta', linewidth=2)
                ax.scatter(mlong, mcol, [10]*len(MLT_IE_prime), color='navy', linewidth=2)
            
            # Quiver plot for flow arrows
            if i < 5:
                dlat, dlong = 4, 8
                theta = np.array(ie["n_psi"] * np.pi / 180.0 + np.pi / 2)[::dlat, ::dlong]
                r = np.array(ie["n_theta"])[::dlat, ::dlong]
                ux = -np.array(ie["n_uy"])[::dlat, ::dlong]
                uy = np.array(ie["n_ux"])[::dlat, ::dlong]
                ax.quiver(theta, r, ux, uy, scale=10)
        
        fig.text(0.1, 0.9, f"time = {ie.meta['time']}")
        
        if save_name and save_path:
            plot_path = os.path.join(save_path, f"{run_name}_{save_name}/")
            os.makedirs(plot_path, exist_ok=True)
            output_file = os.path.join(plot_path, ie.meta['file'].split('/')[-1][3:-4] + ".png")
            fig.savefig(output_file, dpi=400)
            plt.show()

class PlotKeogram(KeogramData):
    """
    Subclass for rendering Keograms (MLT vs Time grids). Includes internal 
    algorithms for boundary tracking and speed calculation.
    """
    __time_labels = ['2100', '2110', '2120', '2130', '2140', '2150', '2200', '2210', '2220']

    def __init__(self, KeogramData_obj, run_name, mlt_range=[6, 18], time_length=None, boundary=None, PLOT_UX=None):
        super().__init__(KeogramData_obj.data, PLOT_UX)
        self.__MLT_plot_range = mlt_range
        self.__boundary_xlimit = boundary
        self.__time_length = time_length
        self.__run_name = run_name

    def __bgd_vel(self, start_index, duration):
        """Averages the initial time steps to determine a quiescent background velocity."""
        data_length = np.shape(self.data)[0]
        return np.nanmean(self.data[0:data_length, start_index:start_index + duration])

    def __get_boundary(self, background_vel, MLT_boundary_ylimit=None):
        """
        Extracts spatial boundaries across the MLT domain using a Full Width at Half Maximum 
        (FWHM) method relative to the signal peak.
        """
        keogram_mlt_num = int(360 / 24 * (self.__MLT_plot_range[1] - self.__MLT_plot_range[0]))
        mlt_range = np.linspace(self.__MLT_plot_range[0], self.__MLT_plot_range[1], keogram_mlt_num)
        boundary = np.zeros((self.__time_length, 2)) * np.nan
        
        for i in range(self.__time_length):
            temp_data = self.data[:, i] - background_vel
            
            # Constrain search area
            if MLT_boundary_ylimit is not None:
                temp1 = np.argwhere((mlt_range < MLT_boundary_ylimit[1]) & (mlt_range > MLT_boundary_ylimit[0]))[:, 0]
                temp_data_limit = temp_data[temp1]
                max_value = np.nanmax(temp_data_limit)  
                max_loca = np.argwhere(max_value == temp_data_limit)[:, 0][0] + temp1[0]
            else:
                max_value = np.nanmax(temp_data)
                max_loca = np.nanargmax(temp_data)
                
            # Scan left (Dawnward) for the half-max boundary
            for j in range(max_loca):
                if mlt_range[j] < MLT_boundary_ylimit[0]: continue
                if temp_data[j] > max_value / 2:
                    boundary[i, 0] = mlt_range[j]
                    break
                    
            # Scan right (Duskward) for the half-max boundary
            for j in range(1, len(mlt_range) - max_loca - 1):
                if mlt_range[-j] > MLT_boundary_ylimit[1]: continue
                if temp_data[-j] > max_value / 2:
                    boundary[i, 1] = mlt_range[-j]
                    break
                    
            # Special case tracking for the epic simulation run
            if self.__run_name == 'run81_epic' and i < self.__boundary_xlimit[0] + 7:
                for j in range(max_loca):
                    if temp_data[max_loca - j] < max_value / 2:
                        boundary[i, 0] = mlt_range[max_loca - j]
                        break
                for j in range(len(mlt_range) - max_loca - 1):
                    if temp_data[j + max_loca + 1] < max_value / 2:
                        boundary[i, 1] = mlt_range[j + max_loca + 1]
                        break
                        
        if self.__boundary_xlimit is not None:
            interval = self.__boundary_xlimit
            boundary[:interval[0], :] = np.nan
            boundary[interval[1]:, :] = np.nan
            
        return boundary    

    def __speed_fitting(self, boundary, start_index, duration_set, latitude):
        """
        Uses SciKit-Learn's LinearRegression to fit lines to the Dawnside and Duskside
        boundaries, returning the azimuthal expansion velocity in meters per second.
        """
        model = LinearRegression()
        radius = 6371 * np.cos(np.pi * latitude / 180)
        z = np.zeros((self.__time_length, 2)) * np.nan
        
        # Dawnside fitting
        temp = boundary[start_index : start_index + duration_set, 0]
        temp1 = temp[~np.isnan(temp)]
        if len(temp1) > 0:
            x = np.linspace(start_index, start_index + len(temp1), len(temp1))
            model.fit(x.reshape(-1, 1), temp1.reshape(-1, 1))
            z[start_index : start_index + len(temp1), 0] = x * model.coef_.flatten() + model.intercept_
            vel = model.coef_[0][0] * 2 * np.pi * radius / 24 * 1000 / 60
            print(f'Dawnside expansion velocity: {vel:.3f} m/s.')
            
        # Duskside fitting
        temp = boundary[start_index : start_index + duration_set, 1]
        temp1 = temp[~np.isnan(temp)]
        if len(temp1) > 0:
            x = np.linspace(start_index, start_index + len(temp1), len(temp1))
            model.fit(x.reshape(-1, 1), temp1.reshape(-1, 1))
            z[start_index : start_index + len(temp1), 1] = x * model.coef_.flatten() + model.intercept_
            vel = model.coef_[0][0] * 2 * np.pi * radius / 24 * 1000 / 60
            print(f'Duskside expansion velocity: {vel:.3f} m/s.')
            
        return z 

    def plot_keogram_MLT(self, MLT_boundary_ylimit, vnorm=None, save_fig_path=None, save_name=None):
        """Generates the final Matplotlib PColorMesh Keogram."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        x = np.linspace(0, self.__time_length, self.__time_length)
        keogram_mlt_num = int(360 / 24 * (self.__MLT_plot_range[1] - self.__MLT_plot_range[0]))
        y = np.linspace(self.__MLT_plot_range[0], self.__MLT_plot_range[1], keogram_mlt_num)
        xx, yy = np.meshgrid(x, y)
        
        boundary, fitting = None, None
        if self.__boundary_xlimit is not None:
            background_vel = self.__bgd_vel(self.__boundary_xlimit[0] - 5, 5)
            boundary = self.__get_boundary(background_vel, MLT_boundary_ylimit)
            fitting = self.__speed_fitting(boundary, self.__boundary_xlimit[0], self.__boundary_xlimit[1] - self.__boundary_xlimit[0], 78)
            
        if vnorm is None:   
            vnorm = [-1 * np.nanmax(self.data), np.nanmax(self.data)]
            if -np.nanmin(self.data) > np.nanmax(self.data):
                vnorm = [np.nanmin(self.data), -np.nanmin(self.data)]

        im0 = ax.pcolormesh(xx, yy, self.data, vmin=vnorm[0], vmax=vnorm[1], cmap='viridis')
        ax.set_xticks(range(0, self.__time_length, 10), self.__time_labels[0:len(range(0, self.__time_length, 10))])
        
        if fitting is not None:
            ax.plot(x, fitting, color='magenta', linewidth=3)
        if boundary is not None:
            ax.plot(x, boundary, color='red', linewidth=3)
            
            # Auto-save boundary array data
            if self.__run_name in ['run25_ideal', 'run81_epic']:
                out_path = f'/Users/weizhang/Desktop/research/EMIC/20181020/{self.__run_name}/IE/boundary_T32.txt'
                np.savetxt(out_path, boundary)
    
        ax.set_ylabel('MLT [h]')
        ax.set_xlabel('UT [min]')
        fig.colorbar(im0, ax=ax).set_label('Velocity (m/s)')  
 
        if save_fig_path is not None and save_name is not None:
            plt.savefig(os.path.join(save_fig_path, f"{save_name}_V_pole.eps"), format='eps')