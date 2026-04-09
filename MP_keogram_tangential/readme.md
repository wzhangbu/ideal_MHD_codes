Here is a comprehensive `README.md` file tailored to the latest set of scripts you provided (`read_data.py`, `calculation.py`, `plot.py`, and `main.py`). It explains the project's purpose, the role of each file, the configuration toggles, and how to run the pipeline.

You can copy and paste the text block below directly into a `README.md` file in your project folder.

```markdown
# Magnetopause X-Line & Force Balance Analysis Pipeline

## Overview
This repository contains a Python-based analysis pipeline designed to process 3D magnetospheric simulation data (Ideal MHD and EMIC/Multi-fluid). The toolset focuses on the dayside magnetopause, automating the detection of magnetic reconnection X-lines, calculating local plasma force balances (e.g., $J \times B$, magnetic tension, pressure gradients), and generating Time vs. Magnetic Local Time (MLT) Keograms to visualize these dynamics over time.

## Project Structure

The pipeline is divided into four modular scripts:

* **`read_data.py`** (Data Ingestion & Preprocessing)
    * Loads simulation outputs extracted via ParaView (CSV format).
    * Filters out nightside data using an empirical parabolic magnetopause boundary.
    * Interpolates 3D scattered unstructured data onto a structured 2D uniform grid.
    * Calculates the Magnetic Local Time (MLT) for all data points.
    * Translates coordinates from GSM to Solar Magnetic (SM) using `spacepy`.
    * Handles species-specific logic for EMIC (calculating total current $J$ from ion/electron velocities).

* **`calculation.py`** (Physics & Mathematics Core)
    * **X-Line Detection:** Identifies magnetic reconnection regions by finding stagnation points in electron outflow ($U_{ez}$) and pinpointing the exact sign-reversal of the normal magnetic field component ($B_n$).
    * **Force Calculations:** Computes cross products ($U_\perp$, Electric Field), magnetic tension force ($(B \cdot \nabla) B$), and $J \times B$ forces.
    * **Plasma Parameters:** Calculates pressure gradients and local Alfvén velocities along the X-line guide field.

* **`plot.py`** (Visualization Engine)
    * Provides 2D and 3D Matplotlib scatter plotting functions.
    * Replicates ParaView's native "Cool to Warm (Extended)" colormap for 1:1 visual consistency without external dependencies.
    * Contains the `PlotKeogram` class, which uses a Full Width at Half Maximum (FWHM) algorithm to dynamically detect and trace the spatial boundaries of physical phenomena.
    * Generates stacked, multi-panel MLT vs. UT Keogram heatmaps.

* **`main.py`** (Execution Driver)
    * The primary control script. 
    * Loops through a directory of time-series CSV files.
    * Bins the highly-resolved spatial data into an MLT-grid (collapsing the spatial dimensions to extract max/mean values).
    * Routes data through the physics calculators and hands the final arrays to the plotting engine.

## Dependencies

Ensure you have the following Python libraries installed before running the pipeline:

```bash
pip install numpy scipy matplotlib spacepy
```
*Note: `spacepy` is heavily relied upon for the Ticktock time module and coordinate transformations (GSM to SM).*

## Configuration & Usage

All primary configurations are handled at the top of **`main.py`**. Before running the script, open `main.py` and adjust the Global Toggles and file paths to match your local environment and current analysis goals.

### Global Toggles (`main.py`)

* `X_LINE` **(0 or 1)**: 
    * `1`: Isolate and process data *only* at the detected magnetic reconnection X-line.
    * `0`: Process the entire global dayside magnetopause domain.
* `EMIC_index` **(0 or 1)**: 
    * `1`: Expects Multi-fluid/PIC variables (e.g., handles separate ion/electron densities and velocities).
    * `0`: Expects standard Ideal MHD variables.
* `SAVE` **(0 or 1)**: 
    * `1`: Automatically save the generated Matplotlib figures to disk.
    * `0`: Run data generation/calculations only (dry run).
* `SM` **(0 or 1)**: 
    * `1`: Transform coordinate reference frames and vector fields to Solar Magnetic (SM).
    * `0`: Keep data in Geocentric Solar Magnetospheric (GSM).
* `TANGENTIAL` **(0 or 1)**: 
    * `1`: Decompose forces and velocities into their tangential and poleward components relative to the magnetopause boundary.

### Running the Pipeline

Once your configurations and hardcoded directory paths (e.g., `path = '/Users/.../x_line_force/'`) are set, simply execute the main driver from your terminal:

```bash
python main.py
```

## Data Input Requirements
The pipeline expects `.csv` files exported from ParaView. 
* For **Ideal MHD**, ensure variables like `points0` (X), `points1` (Y), `points2` (Z), `bx`, `by`, `bz`, `ux`, `uy`, `uz`, `jx`, `jy`, `jz`, `rho`, and gradients (`gradp0`, `gradbx0`, etc.) are present.
* For **EMIC**, ensure species-specific variables like `rhos0`, `rhos1`, `uxs0`, `uxs1` (electron/ion densities and velocities) are included.
```