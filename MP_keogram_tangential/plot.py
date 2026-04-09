#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization module for Magnetospheric/Ionospheric plotting.
Generates Keograms, identifies phase boundaries dynamically, and handles 
custom 2D and 3D scattering plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from read_data import Data

__Re = 6371

def load_coolwardextend():
    """
    Creates a high-fidelity 'Cool to Warm (Extended)' colormap native to ParaView.
    This guarantees that Matplotlib plots perfectly match the ParaView visual exports 
    without needing external color lookup tables (.npy files).
    """
    # Raw ParaView points defining the gradient: (x_position, R, G, B)
    pv_points = [
        0.0, 0.05639999999999999, 0.05639999999999999, 0.47, 
        0.17159223942480895, 0.24300000000000013, 0.4603500000000004, 0.81, 
        0.2984914818394138, 0.3568143826543521, 0.7450246485363142, 0.954367702893722, 
        0.4321287371255907, 0.6882, 0.93, 0.9179099999999999, 
        0.5, 0.8994959551205902, 0.944646394975174, 0.7686567142818399, 
        0.5882260353170073, 0.957107977357604, 0.8338185108985666, 0.5089156299842102, 
        0.7061412605695164, 0.9275207599610714, 0.6214389091739178, 0.31535705838676426, 
        0.8476395308725272, 0.8, 0.3520000000000001, 0.15999999999999998, 
        1.0, 0.59, 0.07670000000000013, 0.11947499999999994
    ]
    pv_below = [0.0, 0.0, 0.0]  # Color for values below vmin
    pv_above = [0.5, 0.5, 0.5]  # Color for values above vmax

    nodes = []
    # Step through the raw list in chunks of 4 to extract (X, R, G, B)
    for i in range(0, len(pv_points), 4):
        nodes.append((pv_points[i], pv_points[i+1], pv_points[i+2], pv_points[i+3]))
    
    # Normalize X coordinates between 0.0 and 1.0 for Matplotlib
    x_vals = [p[0] for p in nodes]
    min_x, max_x = min(x_vals), max(x_vals)
    span = max_x - min_x
    
    normalized_data = [( (x - min_x) / span, (r, g, b) ) for x, r, g, b in nodes]
    
    # Construct the continuous colormap
    my_cmap = mcolors.LinearSegmentedColormap.from_list("ParaView_Exact", normalized_data)
    my_cmap.set_under(pv_below)
    my_cmap.set_over(pv_above)
    
    return my_cmap

def plotcontour_2d_overlapping(y, z, c, c1, file='./t30.cvs', norm=None, save=None, save_path=None, norm1=None):
    """
    Plots overlapping 2D spatial scatter points. Useful for plotting 
    X-line coordinates (c1) directly on top of background parameter distributions (c).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6)) 
    ax.set_aspect('equal')
    
    # Set symmetric normalization if norm bounds are provided
    vmax = norm if norm is not None else np.nanmax(np.abs(c))
    vmin = -1 * vmax if norm is not None else -np.nanmax(np.abs(c))
    
    colorbar = load_coolwardextend()
    im = ax.scatter(y, z, c=c, vmin=vmin, vmax=vmax, cmap=colorbar)
    
    # Overlay the second dataset (usually a boundary or X-line) using a high z-order
    if norm1 is not None:
        im1 = ax.scatter(y, z, c=c1, s=10, vmin=norm1[0], vmax=norm1[1], cmap='RdGy', zorder=3)
    else:
        im1 = ax.scatter(y, z, c=c1, s=10, vmin=-105, vmax=-100, cmap='RdGy', zorder=3)

    fig.colorbar(im, ax=ax)

    title_text = f"time = {file[3:5]}" + (save if save else "")
    fig.text(0.1, 0.9, title_text, size=15)
     
    if save_path:
        plt.savefig(f"{save_path}{file[3:5]}{save}.png", dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

class PlotKeogram(Data):
    """Visualizer class specialized in producing and annotating MLT/UT Keograms."""
    
    def __init__(self, Data_obj, mlt_range, mlt_tick, time_length, boundary=0):
        super().__init__(Data_obj.data, Data_obj.names, Data_obj.X_LINE, Data_obj.SM)
        self.__mlt_range = mlt_range
        self.__mlt_tick = mlt_tick
        self.__time_length = time_length
        self.__mlt_num = int((mlt_range[1] - mlt_range[0]) / mlt_tick + 1)
        self.__BOUNDARY = boundary

    def __bgd_vel(self, start_index, duration, multi_var=None):
        """Calculates a baseline background velocity for normalization/subtraction."""
        if multi_var is None:
            return np.nanmean(self.data[:, start_index : start_index + duration])
        return np.nanmean(self.data[:, start_index : start_index + duration, multi_var])
    
    def __get_boundary(self, background_vel, MLT_limit=None, multi_var=None, interval=None):
        """
        Extracts the spatial boundary of physical phenomena across MLT domains.
        It uses a Full Width at Half Maximum (FWHM) approach: it finds the peak 
        value of a signal (e.g., Velocity) and traces outward until the signal drops to half.
        """
        mlt_range = np.linspace(self.__mlt_range[0], self.__mlt_range[1], self.__mlt_num)
        boundary = np.zeros((self.__time_length, 2)) * np.nan
        
        for i in range(self.__time_length):
            # Extract data for the current time step and subtract the background noise
            temp_data = self.data[:, i] if len(self.data.shape) < 3 else self.data[:, i, multi_var]
            temp_data = temp_data - background_vel
            
            # Find the peak maximum value and its location within the allowed MLT limits
            if MLT_limit is not None:
                temp1 = np.argwhere((mlt_range < MLT_limit[1]) & (mlt_range > MLT_limit[0]))[:, 0]
                temp_data_limit = temp_data[temp1]
                max_value = np.nanmax(temp_data_limit)  
                max_loca = np.argwhere(max_value == temp_data)[:, 0][0]
            else:
                max_value = np.nanmax(temp_data)
                max_loca = np.nanargmax(temp_data)
                
            # Scan left (dawnward) from the peak until value drops to half max
            for j in range(max_loca):
                if MLT_limit and mlt_range[j] < MLT_limit[0]: continue
                if temp_data[j] < max_value / 2:
                    boundary[i, 0] = mlt_range[j]
                    break
            
            # Scan right (duskward) from the peak until value drops to half max
            for j in range(1, len(mlt_range) - max_loca - 1):
                if MLT_limit and mlt_range[-j] > MLT_limit[1]: continue
                if temp_data[-j] < max_value / 2:
                    boundary[i, 1] = mlt_range[-j]
                    break

        # Wipe out boundaries in specified gap intervals (e.g., if tracing fails)
        if interval is not None:
            boundary[:interval[0], :] = np.nan
            boundary[interval[1]:, :] = np.nan

        return boundary

    def _plot_keogram_MLT_GEM(self, names, MLT_limit, vnorm=0, boundary_nan=None, save_path=None, IE_tracing=None, EMIC_index=0, plot_num=3):
        """
        Generates a stacked multi-panel plot comparing different variable Keograms.
        Produces high-quality 2D heatmaps with Time on X and MLT on Y.
        """
        y_keogram = self.data
        tplotnames = self.names
        
        # Build 2D mesh grid for the heatmap
        x = np.linspace(25, 25 + self.__time_length, self.__time_length)
        y = np.linspace(self.__mlt_range[0], self.__mlt_range[1], self.__mlt_num)
        xx, yy = np.meshgrid(x, y)
            
        # Iterate through the variables to plot in groups defined by `plot_num`
        for i in range(int(len(names) / plot_num)):
            name_index = [tplotnames.index(names[plot_num*i + j]) for j in range(plot_num)]
            
            fig, axs = plt.subplots(plot_num, 1, figsize=(6, 3 * plot_num))
            fig.tight_layout()
            plt.subplots_adjust(left=0.15, bottom=0.1, right=0.99, top=0.99, hspace=0.1)
            
            for k, ax, title in zip(range(plot_num), axs, names[plot_num * i : plot_num*i + plot_num]):
                dd = y_keogram[..., name_index[k]]
                norm = [-1 * np.nanmax(np.abs(dd)), np.nanmax(np.abs(dd))] if vnorm == 0 else [-250, 250]
                
                # Plot the underlying colormesh heatmap
                im0 = ax.pcolormesh(xx, yy, abs(dd), vmin=norm[0], vmax=norm[1], cmap='viridis')   

                # Overlay tracing limits if provided
                if IE_tracing is not None: 
                    ax.plot(x, IE_tracing, color='aqua', linewidth=6)

                # Overlay dynamically calculated FWHM boundaries
                if self.__BOUNDARY == 1:
                    background_vel = self.__bgd_vel(0, 3, 3*i + k)
                    boundary = self.__get_boundary(background_vel, MLT_limit=MLT_limit, multi_var=3*i + k, interval=boundary_nan)
                    ax.plot(x, boundary, color='blue')
                
                # Formatting
                ax.set_xticks([])
                if k == plot_num - 1:
                    ax.set_xlabel('UT [min]')
                    ax.set_xticks(range(25, 25 + self.__time_length, 5))
                
                ax.set_ylabel(f'{title}\n MLT [h]')
                fig.colorbar(im0, ax=ax)  

            if save_path is not None:
                plt.savefig(f"{save_path}{tplotnames[3*i]}.png")
            plt.close(fig)