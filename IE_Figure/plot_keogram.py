#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting module to visualize Convection maps and Keograms.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import spacepy.pybats.rim as rim
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import IE_Figure.calculation as calculation
from IE_Figure.calculation import KeogramData

def convection(file_path, run_name, start_index, file_num, IE_OCB=None, save_path=None, save_name=None):
    """
    Generates and plots ionospheric convection maps from PyBats IDL files.
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
    print(f'There are {len(files)} files in total.')
    
    for k, filename in enumerate(files):
        if k < start_index or k > (start_index + file_num):
            continue

        print(f"Plotting: {filename}")
        fig = plt.figure(figsize=(8, 12))
        ie = rim.Iono(filename)
        
        ie["n_r"] = np.sqrt(ie["n_x"]**2 + ie["n_y"]**2)
        ie["V_pole"] = - (ie["n_ux"] * ie["n_x"] + ie["n_uy"] * ie["n_y"]) / ie["n_r"]        
        ie["n_u"] = np.sqrt(ie["n_ux"]**2 + ie["n_uy"]**2 + ie["n_uz"]**2) 
        
        vars_to_plot = ["V_pole", "n_ux"]
        latrange = 18

        time = int(ie.meta['file'].split('/')[-1][9:13])
        if time < IE_start:
            MLT_IE, MLAT_IE = MLT_IE_all[:, 0], MLAT_IE_all[:, 0]
        elif time > IE_end:
            MLT_IE, MLAT_IE = MLT_IE_all[:, -1], MLAT_IE_all[:, -1]
        else:
            MLT_IE, MLAT_IE = MLT_IE_all[:, time-IE_start], MLAT_IE_all[:, time-IE_start]
        
        valid = ~(np.isnan(MLT_IE) | np.isnan(MLAT_IE))
        mlt_origin, mcol_origin = MLT_IE[valid], 90. - MLAT_IE[valid]
        mlong_origin = ((mlt_origin - 12) * 15 + 90) / 180. * np.pi
        
        MLT_IE_prime, MLAT_IE_prime = calculation.fit_OCB(MLT_IE, MLAT_IE)           
        mcol = 90. - MLAT_IE_prime
        mlong = ((MLT_IE_prime - 12) * 15 + 90) / 180. * np.pi
        
        for i, var in enumerate(vars_to_plot):
            loc = 210 + 1 + i
            if var == "V_pole":
                maxz, cmap = 0.6, 'viridis'
            elif var == "n_u":
                maxz, cmap = 0.65, 'jet'
            elif var == "n_ux":
                maxz, cmap = 0.9, 'bwr'
            else:
                maxz, cmap = 15, 'jet'
                
            fig, ax, cnt, cb = ie.add_cont(var, add_cbar=True, target=fig, loc=loc, lines=False, minz=0, maxz=maxz, cmap=cmap, extend='both')
            
            cnt.cmap.set_under(color='black' if cmap == 'viridis' else 'blue')
            cnt.cmap.set_over(color='yellow' if cmap == 'viridis' else 'darkred')
            cb.set_label('m/s')
            ax.set_ylim(0, latrange)
            
            # Plot reference scatter markers
            z_origin = [15] * len(mlt_origin)
            ax.scatter(mlong_origin, mcol_origin, z_origin, color='magenta', linewidth=2)
            z_prime = [10] * len(MLT_IE_prime)
            ax.scatter(mlong, mcol, z_prime, color='navy', linewidth=2)
            
            if i < 5:
                dlat, dlong = 4, 8
                theta = np.array(ie["n_psi"] * np.pi / 180.0 + np.pi / 2)[::dlat, ::dlong]
                r = np.array(ie["n_theta"])[::dlat, ::dlong]
                ux = -np.array(ie["n_uy"])[::dlat, ::dlong]
                uy = np.array(ie["n_ux"])[::dlat, ::dlong]
                ax.quiver(theta, r, ux, uy, scale=10)
        
        fig.text(0.1, 0.9, "time = " + str(ie.meta['time']))
        
        if save_name and save_path:
            plot_path = os.path.join(save_path, f"{run_name}_{save_name}/")
            os.makedirs(plot_path, exist_ok=True)
            output_file = os.path.join(plot_path, ie.meta['file'].split('/')[-1][3:-4] + ".png")
            fig.savefig(output_file, dpi=400)
            print(f'Saved successfully at {plot_path}')
            plt.show()


class PlotKeogram(KeogramData):
    """
    Subclass of KeogramData dedicated to generating visual Keogram plots 
    and estimating boundary expansion velocities.
    """
    __time_labels = ['2100', '2110', '2120', '2130', '2140', '2150', '2200', '2210', '2220']

    def __init__(self, keogram_data_obj, run_name, mlt_range=[6, 18], time_length=None, boundary=None, PLOT_UX=None):
        super().__init__(keogram_data_obj.data, PLOT_UX)
        self.__MLT_plot_range = mlt_range
        self.__boundary_xlimit = boundary
        self.__time_length = time_length
        self.__run_name = run_name

    def __bgd_vel(self, start_index, duration):
        """Calculates background velocity mean to subtract from signals."""
        data_length = np.shape(self.data)[0]
        vel = np.nanmean(self.data[0:data_length, start_index : start_index + duration])
        return vel    
    
    def plot_keogram_MLT(self, MLT_boundary_ylimit, vnorm=None, save_fig_path=None, save_name=None):
        """Plots the primary MLT Keogram mesh and identified boundaries."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        x = np.linspace(0, self.__time_length, self.__time_length)
        keogram_mlt_num = int(360 / 24 * (self.__MLT_plot_range[1] - self.__MLT_plot_range[0]))
        y = np.linspace(self.__MLT_plot_range[0], self.__MLT_plot_range[1], keogram_mlt_num)
        xx, yy = np.meshgrid(x, y)
        
        boundary = None
        fitting = None
        if self.__boundary_xlimit is not None:
            background_vel = self.__bgd_vel(self.__boundary_xlimit[0] - 5, 5)
            boundary = self.__get_boundary(background_vel, MLT_boundary_ylimit)
            fitting = self.__speed_fitting(boundary, self.__boundary_xlimit[0], self.__boundary_xlimit[1] - self.__boundary_xlimit[0], 78)
            
        if vnorm is None:   
            vnorm = [-1 * np.nanmax(self.data), np.nanmax(self.data)]
            if -1 * np.nanmin(self.data) > np.nanmax(self.data):
                vnorm = [np.nanmin(self.data), -1 * np.nanmin(self.data)]

        im0 = ax.pcolormesh(xx, yy, self.data, vmin=vnorm[0], vmax=vnorm[1], cmap='viridis')
        ax.set_xticks(range(0, self.__time_length, 10), self.__time_labels[0 : len(range(0, self.__time_length, 10))])
        
        if fitting is not None:
            ax.plot(x, fitting, color='magenta', linewidth=3)
        if boundary is not None:
            ax.plot(x, boundary, color='red', linewidth=3)
            print('The boundary is:\n', np.round(boundary, 3))
            
            # Dump boundries if ideal or epic
            base_p = '/Users/weizhang/Desktop/research/EMIC/20181020/'
            if self.__run_name in ['run25_ideal', 'run81_epic']:
                np.savetxt(os.path.join(base_p, f"{self.__run_name}/IE/boundary_T32.txt"), boundary)
    
        ax.set_ylabel('MLT [h]')
        ax.set_xlabel('UT [min]')
        fig.colorbar(im0, ax=ax).set_label('Velocity (m/s)')  
 
        if save_fig_path is not None and save_name is not None:
            plt.savefig(os.path.join(save_fig_path, f"{save_name}_V_pole.eps"), format='eps')
            
    def __get_boundary(self, background_vel, MLT_boundary_ylimit=None):
        """Identifies boundary signals based on Full Width at Half Maximum relative to velocity peaks."""
        keogram_mlt_num = int(360 / 24 * (self.__MLT_plot_range[1] - self.__MLT_plot_range[0]))
        mlt_range = np.linspace(self.__MLT_plot_range[0], self.__MLT_plot_range[1], keogram_mlt_num)
        
        boundary = np.zeros((self.__time_length, 2)) * np.nan
        for i in range(self.__time_length):
            temp_data = self.data[:, i] - background_vel
            
            if MLT_boundary_ylimit is not None:
                temp1 = np.argwhere((mlt_range < MLT_boundary_ylimit[1]) & (mlt_range > MLT_boundary_ylimit[0]))[:, 0]
                temp_data_limit = temp_data[temp1]
                max_value = np.nanmax(temp_data_limit)  
                max_loca = np.argwhere(max_value == temp_data_limit)[:, 0][0] + temp1[0]

            # Trace boundary from two sides to max
            for j in range(max_loca):
                if mlt_range[j] < MLT_boundary_ylimit[0]:
                    continue
                if temp_data[j] > max_value / 2:
                    boundary[i, 0] = mlt_range[j]
                    break
            for j in range(1, len(mlt_range) - max_loca - 1):
                if mlt_range[-j] > MLT_boundary_ylimit[1]:
                    continue
                if temp_data[-j] > max_value / 2:
                    boundary[i, 1] = mlt_range[-j]
                    break
                
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
        """Fits a linear regression model to the boundaries to compute expansion speeds."""
        model = LinearRegression()
        radius = 6371 * np.cos(np.pi * latitude / 180)
        z = np.zeros((self.__time_length, 2)) * np.nan
        
        # Dawnside expansion
        temp = boundary[start_index : start_index + duration_set, 0]
        temp1 = temp[~np.isnan(temp)]
        duration = len(temp1)
        if duration > 0:
            x = np.linspace(start_index, start_index + duration, duration)
            model.fit(x.reshape(-1, 1), temp1.reshape(-1, 1))
            z[start_index : start_index + duration, 0] = x * model.coef_.flatten() + model.intercept_
            vel = model.coef_[0][0] * 2 * np.pi * radius / 24 * 1000 / 60
            print(f'The dawnside expansion velocity is {float(np.around(vel, 3))} m/s.')
        
        # Duskside expansion
        temp = boundary[start_index : start_index + duration_set, 1]
        temp1 = temp[~np.isnan(temp)]
        duration = len(temp1)
        if duration > 0:
            x = np.linspace(start_index, start_index + duration, duration)
            model.fit(x.reshape(-1, 1), temp1.reshape(-1, 1))
            z[start_index : start_index + duration, 1] = x * model.coef_.flatten() + model.intercept_
            vel = model.coef_[0][0] * 2 * np.pi * radius / 24 * 1000 / 60
            print(f'The duskside expansion velocity is {float(np.around(vel, 3))} m/s.')
            
        return z