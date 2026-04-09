# Ionospheric Electrodynamics & OCB Analysis Pipeline

## Overview
This project contains a suite of Python and ParaView scripts designed to analyze Ionosphere Electrodynamics (IE) data and track the Open-Closed field line Boundary (OCB). It processes BATSRUS/SWMF output files to generate Keograms, plot ionospheric convection maps, and utilize 3D magnetic field line tracing via ParaView to identify open and closed topological boundaries.

## Repository Structure

The pipeline is split into three main categories: Core Processing, Visualization, and Boundary Tracing.

### Core Processing & Visualization
* **`calculation.py`**: The core data processing module. It handles reading IE data, calculating coordinate transformations (MLT/MLAT to Cartesian SM/GSM), and includes the `KeogramData` class for extracting poleward velocities across time-series data.
* **`plot_keogram.py`**: The primary visualization engine. It contains the `PlotKeogram` subclass to generate MLT vs. UT Keogram heatmaps and perform linear regression to estimate boundary expansion velocities. It also includes functions to plot 2D ionospheric convection maps.
* **`main.py`**: The main execution driver for the Keogram and Convection mapping pipeline. This is where you configure your event dates, run names (e.g., `run25_ideal`, `run81_epic`), and directory paths before generating the final plots.

### OCB & Field Line Tracing
* **`OCB_in_PV.py`**: A ParaView Python (`pvpython`) script. It automates loading 3D Global Magnetosphere (GM) `.vtm` files, creates spherical seed geometries, and traces magnetic field lines backward to determine their termination status (Open vs. Closed). It exports CSVs of the tracing data and visual screenshots.
* **`OCB_python.py`**: A post-processing script for the ParaView CSV outputs. It uses step-function minimization to find the exact boundaries where field lines transition from closed to open, converts coordinates using


2. Process Tracing Data
Run the post-processing script to extract the OCB from the ParaView CSVs.

Bash
python OCB_python.py
Output: Text files containing MLAT/MLT coordinates of the Open-Closed Boundary.

3. Generate Keograms & Convection Maps
Open main.py and update the run_name, base_dir, and save_path variables to match your current event and local directory structure. Execute the main driver.

Bash
python main.py
Output: Generates Keogram .eps plots and/or Convection map .png arrays.

4. Cross-Reference Boundaries (Optional)
If you need to strictly match the visually identified boundaries from the Keograms to exact physical coordinates in the simulation data, run:

Bash
python OCB_in_keogram.py
Configuration Notes
Directory paths (e.g., /Users/weizhang/Desktop/research/EMIC/...) and event-specific configurations are currently defined at the top or bottom of each script. Before running the pipeline on a new machine or for a new simulation event, ensure you update the base_dir, run_name, and EVENT_INDEX variables across main.py, OCB_python.py, and OCB_in_PV.py.