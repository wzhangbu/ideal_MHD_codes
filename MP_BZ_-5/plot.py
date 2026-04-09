#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization module for Magnetospheric plotting.
Generates Keograms, identifies phase boundaries dynamically, and handles 
custom 2D and 3D scattering plots.
"""

import numpy as np
from read_data import Data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plotcontour_3d(x, y, z, c, file='./t30.csv', norm=None, save=None, save_path=None):
    """Plots 3D spatial scatter data with a coolwarm colormap."""
    fig = plt.figure(figsize=(10, 8)) 
    ax = plt.axes(projection='3d')
    
    if norm is None:
        vmax = np.nanmax(c)
        vmin = np.nanmin(c)
    else:
        vmin = -1 * norm
        vmax = norm

    # delete NaN values to prevent plotting errors
    c_clean = c[~np.isnan(c)]
    
    im = ax.scatter(x, y, z, c=c, vmin=vmin, vmax=vmax, cmap='coolwarm')
    fig.colorbar(im, ax=ax)

    title_suffix = save if save else ""
    fig.text(0.1, 0.9, f"time = {file[3:5]} {title_suffix}", size=15)

    ax.set(xlabel='X', ylabel='Y', zlabel='Z')
    # View the XY plane primarily
    ax.view_init(180, 0) 
    
    if save_path is not None:
        plt.savefig(save_path + file[3:5] + save + '.png')

    plt.show()
    
def plotcontour_2d(y, z, c, file='./t30.csv', norm=None, save=None, save_path=None):
    """Plots 2D spatial scatter data on a flat projection."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6)) 
    
    if norm is None:
        vmax = np.nanmax(np.abs(c))
        vmin = -1 * vmax
    else:
        vmin = -1 * norm
        vmax = norm

    im = ax.scatter(y, z, c=c, vmin=vmin, vmax=vmax, cmap='coolwarm')
    fig.colorbar(im, ax=ax)

    title_suffix = save if save else ""
    fig.text(0.1, 0.9, f"time = {file[3:5]} {title_suffix}", size=15)
     
    if save_path is not None:
        plt.savefig(save_path + file[3:5] + save + '.png')

    plt.show()

class PlotKeogram(Data):
    """Visualizer class specialized in producing and annotating MLT/UT Keograms."""
    
    def __init__(self, Data_obj, mlt_range=None, mlt_tick=None, time_length=None, boundary=None):
        super(PlotKeogram, self).__init__(Data_obj.data, Data_obj.names, Data_obj.X_LINE, Data_obj.SM)
        
        self.__mlt_range = mlt_range or [6, 18]
        self.__mlt_tick = mlt_tick or 0.1
        self.__time_length = time_length or 0
        self.__mlt_num = int((self.__mlt_range[1] - self.__mlt_range[0]) / self.__mlt_tick + 1)
        self.__BOUNDARY = boundary

    def __bgd_vel(self, start_index, duration, multi_var=None):
        """Calculates a baseline background velocity across the initial time steps to normalize boundaries."""
        data_length = np.shape(self.data)[0]
        init, end = 0, data_length
        if multi_var is None:
            vel = np.nanmean(self.data[init:end, start_index:start_index + duration])
        else:
            vel = np.nanmean(self.data[init:end, start_index:start_index + duration, multi_var])
        return vel 
    
    def __get_boundary(self, background_vel, MLT_limit=None, multi_var=None, interval=None):
        """
        Extracts the spatial boundary of physical phenomena across MLT domains.
        It uses a Full Width at Half Maximum (FWHM) approach: it finds the peak 
        value of a signal (e.g., Velocity) and traces outward until the signal drops to half.
        """
        mlt_range = np.linspace(self.__mlt_range[0], self.__mlt_range[1], self.__mlt_num)
        
        if MLT_limit is not None:
            temp = np.where((mlt_range < MLT_limit[1]) & (mlt_range > MLT_limit[0]))
            data_temp = self.data.reshape(1, np.size(self.data))
            max_val = np.max(data_temp[0, temp], axis=1)
            max_loca = np.argwhere(data_temp == max_val)[0, 1]

        boundary = np.zeros((self.__time_length, 2)) * np.nan
        
        for i in range(self.__time_length):
            idx_var = -5 if multi_var == 2 else multi_var
            
            # Isolate the data slice for the current timestep
            if len(np.shape(self.data)) < 3:
                temp_data = self.data[:, i]
            else:
                temp_data = self.data[:, i, idx_var]    
                
            temp_data = temp_data - background_vel
            
            if MLT_limit is not None:
                temp1 = np.argwhere((mlt_range < MLT_limit[1]) & (mlt_range > MLT_limit[0]))[:, 0]
                temp_data_limit = temp_data[temp1]
                max_value = np.nanmax(temp_data_limit)  
                max_loca = np.argwhere(max_value == temp_data)[:, 0][0]
                
            # Scan left (dawnward) from the peak until value drops to half max
            for j in range(max_loca):
                if mlt_range[j] < MLT_limit[0]: continue
                if temp_data[j] < max_value / 2:
                    boundary[i, 0] = mlt_range[j]
                    break
            
            # Scan right (duskward) from the peak until value drops to half max
            for j in range(1, len(mlt_range) - max_loca - 1):
                if mlt_range[-j] > MLT_limit[1]: continue
                if temp_data[-j] < max_value / 2:
                    boundary[i, 1] = mlt_range[-j]
                    break

        if interval is not None:
            boundary[:interval[0], :] = np.nan
            boundary[interval[1]:, :] = np.nan

        return boundary
    
    def _plot_keogram_MLT_GEM(self, names, MLT_limit, vnorm=0, boundary_nan=None, save_path=None, IE_tracing=None, EMIC_index=0, small_zlim=0, plot_num=3):
        """
        Generates a stacked multi-panel plot comparing different variable Keograms.
        Produces high-quality 2D heatmaps with Time on X and MLT on Y.
        """
        y_keogram, tplotnames = self.data, self.names
        
        # Build 2D mesh grid for the heatmap
        x = np.linspace(25, 25 + self.__time_length, self.__time_length)
        y = np.linspace(self.__mlt_range[0], self.__mlt_range[1], self.__mlt_num)
        xx, yy = np.meshgrid(x, y)
            
        for i in range(int(len(names)/plot_num)):
            name_index = [tplotnames.index(names[plot_num*i+j]) for j in range(plot_num)]
                
            fig, axs = plt.subplots(plot_num, 1, figsize=(6, 3* plot_num))
            fig.tight_layout()
            plt.subplots_adjust(left=0.15, bottom=0.1, right=0.99, top=0.99, hspace=0.1)
            
            for k, ax, dd, title in zip(range(plot_num), axs, [y_keogram[..., name_index[j]] for j in range(plot_num)], names[plot_num * i: plot_num*i + plot_num]):

                if vnorm == 0:
                    norm = [-1 * np.nanmax(np.abs(dd)), np.nanmax(np.abs(dd))]
                else:
                    # Provide standard fallbacks depending on parameter type
                    if title == 'uperpx' or title == 'ueperpx': norm = [0, 250]
                    elif title == 'ux': norm = [-250, 250]
                    else: norm = [-0.15, 0.15]
                
                # Render heatmap
                im0 = ax.pcolormesh(xx, yy, dd, vmin=norm[0], vmax=norm[1], cmap='gnuplot2')   

                # Overlay tracing limits if provided
                if IE_tracing is not None and np.nanmax(IE_tracing) is not None: 
                    ax.plot(x, IE_tracing, color='aqua', linewidth=6)

                # Overlay dynamically calculated FWHM boundaries
                if self.__BOUNDARY == 1:
                    background_vel = self.__bgd_vel(0, 3, 3*i + k)
                    boundary = self.__get_boundary(background_vel, MLT_limit=MLT_limit, multi_var=3*i + k, interval=boundary_nan)
                    ax.plot(x, boundary, color='blue')
                
                # Prettify labels based on scientific standard
                Force_label = r'$10^{-15}N$'
                title_map = {
                    'gradp0': (r'$\nabla P_{th}$ in X', Force_label),
                    'JxBx': (r'$JxB$ in X', Force_label),
                    'Ftotalx': (r'$F_{total}$ in X', Force_label),
                    'Tensionx': ('Tension Force in X', Force_label),
                    'gradpb0': (r'$\nabla P_{mag}$ in X', Force_label),
                    'uperpx': (r'$U_\perp$ in X', 'km/s'),
                    'upole': (r'$U_{Pole}$ in SM', 'km/s'),
                    'Va': (r'$V_{Alfven}$ in X', 'km/s')
                }
                title, label = title_map.get(title, (title, 'km/s' if 'u' in title.lower() else Force_label))

                ax.set_xticks([])
                if k == plot_num - 1:
                    ax.set_xlabel('UT [min]')
                    ax.set_xticks(range(25, 25+self.__time_length, 5))
                ax.set_ylabel(title + '\n MLT [h]')
                fig.colorbar(im0, ax=ax).set_label(label)  

            if save_path is not None:
                plt.savefig(save_path + tplotnames[3*i] + '.png')