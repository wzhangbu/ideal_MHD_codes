#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main execution script to configure events, ranges, and generate Keograms.
"""

from IE_Figure.calculation import KeogramData
from IE_Figure.plot_keogram import PlotKeogram
import os

# --- Configurations ---
run_name = 'run81_epic'
base_dir = '/Users/weizhang/Desktop/research/EMIC'
save_path = os.path.join(base_dir, 'Figure/20181020/maps/north_hemisphere/')
path = os.path.join(base_dir, f'20181020/{run_name}/IE/')

mlt_cal_range = [6, 18]
mlt_plot_range = [6, 18]   
plot_ux = None

# --- Event Triggers ---
if run_name == 'run25_ideal':
    file_num, start_index, wide_range = 59, 0, 0 
    boundary_xlimit = [34, 52] 
    boundary_ylimit = [8, 14]
elif run_name == 'run26_hall':
    file_num, start_index, wide_range = 59, 0, 0   
    boundary_xlimit = [35, 51]
    boundary_ylimit = [8, 14]
elif run_name == 'run31_epic':
    file_num, start_index, wide_range = 59, 0, 0     
    boundary_xlimit = [35, 52]
    boundary_ylimit = [8, 16]
elif run_name == 'run80_epic':
    file_num, start_index, wide_range = 59, 0, 0     
    boundary_xlimit = [34, 49]
    boundary_ylimit = [8, 16]    
elif run_name == 'run81_epic':
    file_num, start_index, wide_range = 59, 0, 0     
    boundary_xlimit = [32, 50]
    boundary_ylimit = [8, 16]    
    mlt_cal_range = [8, 16]
    mlt_plot_range = [8, 16]   

# --- Execution ---
if __name__ == "__main__":
    keogram_data = KeogramData()
    keogram_data.read_from_file(path, file_num, start_index, run_name, wide_range, plot_ux, mlt_cal_range)
    
    keogram_plotter = PlotKeogram(keogram_data, run_name, mlt_range=mlt_plot_range, 
                                  time_length=file_num, boundary=boundary_xlimit, PLOT_UX=plot_ux)
                                  
    keogram_plotter.plot_keogram_MLT(boundary_ylimit, vnorm=[0, 600], 
                                     save_fig_path=save_path, save_name=run_name)