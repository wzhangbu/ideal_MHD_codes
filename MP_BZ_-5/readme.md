Here is a comprehensive `README.md` file tailored to your complete Magnetosphere-Ionosphere analysis project, formatted in standard Markdown. You can copy and paste this directly into your repository.

```markdown
# Magnetosphere-Ionosphere Electrodynamics & OCB Analysis Pipeline
# Magnetosphere-Ionosphere Electrodynamics & OCB Analysis Pipeline

## Overview
This repository contains a comprehensive suite of Python and ParaView scripts designed to analyze 3D magnetospheric simulation data (Ideal MHD and EMIC/Multi-fluid models like SWMF/BATSRUS). The pipeline automates the detection of magnetic reconnection X-lines, calculates local plasma force balances (e.g., $J \times B$, magnetic tension, pressure gradients), traces magnetic field lines to identify the Open-Closed field line Boundary (OCB), and generates Time vs. Magnetic Local Time (MLT) Keograms and Ionospheric Convection maps.

## Features
* **Magnetic Topology Tracing:** Automates 3D field line tracing via ParaView to determine open vs. closed field line topologies.
* **X-Line Detection:** Pinpoints dayside magnetic reconnection X-lines using $B_n$ sign reversals and electron outflow ($U_{ez}$) stagnation points.
* **Plasma Force Calculations:** Computes cross products, magnetic tension forces, pressure gradients, and Alfvén velocities.
* **Coordinate Transformations:** Converts between Geocentric Solar Magnetospheric (GSM) and Solar Magnetic (SM) coordinates using `spacepy` and custom dipole models.
* **Advanced Visualization:** Generates stacked Keogram heatmaps, 2D/3D scatter plots, and PyBats-driven ionospheric convection maps with dynamically tracked boundaries (using FWHM algorithms and linear regression).

---

## Repository Structure

### 1. Main Execution & Drivers
* **`main.py`**: The primary control script. Iterates through temporal CSV datasets, bins high-resolution spatial data into an MLT-grid, computes variables across the magnetopause, and triggers Keogram plotting.
* **`test.py`**: A lightweight script for testing coordinate transformations and dipole tilt functions.

### 2. Core Processing & Physics
* **`read_data.py`**: Handles data ingestion (CSV format). Filters nightside data using empirical magnetopause boundaries, interpolates 3D unstructured data onto a 2D uniform grid, and manages GSM-to-SM conversions.
* **`calculation.py` / `calc.py`**: The physics and mathematics core. Contains functions for extracting cross products ($U_\perp$, Electric Field), magnetic tension force ($(B \cdot \nabla) B$), and identifying X-lines. Prepares data fields for Keogram binning.
* **`dipole.py`**: A robust utility class for calculations involving the Earth's magnetic dipole. Handles IGRF coefficient interpolation, dipole tilt calculation, apex coordinate base vectors, and magnetic flux integration.

### 3. Open-Closed Boundary (OCB) Tracing
* **`OCB_in_PV.py`**: A ParaView Python (`pvpython`) macro. Automates loading Global Magnetosphere (GM) `.vtm` files, creates spherical seed geometries, traces field lines backward, and exports termination status CSVs.
* **`OCB_python.py` / `OCB_GM.py`**: Post-processing scripts for the ParaView trace exports. Uses step-function minimization to find the exact boundary where field lines transition from closed to open, converting the results into MLAT/MLT arrays.
* **`OCB_at_9RE.py`**: Extracts boundary conditions specifically at a fixed radial distance in the tail/flank (e.g., 9.5 $R_E$).
* **`OCB_in_keogram.py`**: Cross-references the boundaries identified in visual Keograms with the exact MLAT/MLT coordinates from Ionosphere Electrodynamics (IE) data files.

### 4. Visualization Engine
* **`plot_keogram.py`**: Generates MLT vs. UT Keogram heatmaps and 2D ionospheric convection maps from `.idl` files. Includes the `PlotKeogram` class which performs linear regression to estimate azimuthal boundary expansion speeds.
* **`plot.py`**: Provides 2D and 3D Matplotlib scatter plotting functions. Replicates ParaView's native "Cool to Warm (Extended)" colormap for 1:1 visual consistency.

---

## Dependencies
Ensure you have the following software and Python libraries installed:
* **Python 3.x**
* **ParaView 5.11+** (The bundled `pvpython` executable is required for `OCB_in_PV.py`)
* `numpy`
* `scipy` (for grid interpolation and curve fitting)
* `matplotlib`
* `scikit-learn` (for boundary linear regression)
* `spacepy` (specifically `spacepy.pybats.rim` and `spacepy.coordinates`)
* `pandas` (required by `dipole.py` for datetime indexing)

---

## Configuration & Standard Workflow

### Step 1: 3D Field Line Tracing (ParaView)
Ensure your `.vtm` simulation files are in the designated GM directory. Run the tracing script using ParaView's python environment to generate the trace data.
```bash
pvpython OCB_in_PV.py

## Overview
This repository contains a comprehensive suite of Python and ParaView scripts designed to analyze 3D magnetospheric simulation data (Ideal MHD and EMIC/Multi-fluid models like SWMF/BATSRUS). The pipeline automates the detection of magnetic reconnection X-lines, calculates local plasma force balances (e.g., $J \times B$, magnetic tension, pressure gradients), traces magnetic field lines to identify the Open-Closed field line Boundary (OCB), and generates Time vs. Magnetic Local Time (MLT) Keograms and Ionospheric Convection maps.

## Features
* **Magnetic Topology Tracing:** Automates 3D field line tracing via ParaView to determine open vs. closed field line topologies.
* **X-Line Detection:** Pinpoints dayside magnetic reconnection X-lines using $B_n$ sign reversals and electron outflow ($U_{ez}$) stagnation points.
* **Plasma Force Calculations:** Computes cross products, magnetic tension forces, pressure gradients, and Alfvén velocities.
* **Coordinate Transformations:** Converts between Geocentric Solar Magnetospheric (GSM) and Solar Magnetic (SM) coordinates using `spacepy` and custom dipole models.
* **Advanced Visualization:** Generates stacked Keogram heatmaps, 2D/3D scatter plots, and PyBats-driven ionospheric convection maps with dynamically tracked boundaries (using FWHM algorithms and linear regression).

---

## Repository Structure

### 1. Main Execution & Drivers
* **`main.py`**: The primary control script. Iterates through temporal CSV datasets, bins high-resolution spatial data into an MLT-grid, computes variables across the magnetopause, and triggers Keogram plotting.
* **`test.py`**: A lightweight script for testing coordinate transformations and dipole tilt functions.

### 2. Core Processing & Physics
* **`read_data.py`**: Handles data ingestion (CSV format). Filters nightside data using empirical magnetopause boundaries, interpolates 3D unstructured data onto a 2D uniform grid, and manages GSM-to-SM conversions.
* **`calculation.py` / `calc.py`**: The physics and mathematics core. Contains functions for extracting cross products ($U_\perp$, Electric Field), magnetic tension force ($(B \cdot \nabla) B$), and identifying X-lines. Prepares data fields for Keogram binning.
* **`dipole.py`**: A robust utility class for calculations involving the Earth's magnetic dipole. Handles IGRF coefficient interpolation, dipole tilt calculation, apex coordinate base vectors, and magnetic flux integration.

### 3. Open-Closed Boundary (OCB) Tracing
* **`OCB_in_PV.py`**: A ParaView Python (`pvpython`) macro. Automates loading Global Magnetosphere (GM) `.vtm` files, creates spherical seed geometries, traces field lines backward, and exports termination status CSVs.
* **`OCB_python.py` / `OCB_GM.py`**: Post-processing scripts for the ParaView trace exports. Uses step-function minimization to find the exact boundary where field lines transition from closed to open, converting the results into MLAT/MLT arrays.
* **`OCB_at_9RE.py`**: Extracts boundary conditions specifically at a fixed radial distance in the tail/flank (e.g., 9.5 $R_E$).
* **`OCB_in_keogram.py`**: Cross-references the boundaries identified in visual Keograms with the exact MLAT/MLT coordinates from Ionosphere Electrodynamics (IE) data files.

### 4. Visualization Engine
* **`plot_keogram.py`**: Generates MLT vs. UT Keogram heatmaps and 2D ionospheric convection maps from `.idl` files. Includes the `PlotKeogram` class which performs linear regression to estimate azimuthal boundary expansion speeds.
* **`plot.py`**: Provides 2D and 3D Matplotlib scatter plotting functions. Replicates ParaView's native "Cool to Warm (Extended)" colormap for 1:1 visual consistency.

---

## Dependencies
Ensure you have the following software and Python libraries installed:
* **Python 3.x**
* **ParaView 5.11+** (The bundled `pvpython` executable is required for `OCB_in_PV.py`)
* `numpy`
* `scipy` (for grid interpolation and curve fitting)
* `matplotlib`
* `scikit-learn` (for boundary linear regression)
* `spacepy` (specifically `spacepy.pybats.rim` and `spacepy.coordinates`)
* `pandas` (required by `dipole.py` for datetime indexing)

---

## Configuration & Standard Workflow

### Step 1: 3D Field Line Tracing (ParaView)
Ensure your `.vtm` simulation files are in the designated GM directory. Run the tracing script using ParaView's python environment to generate the trace data.
```bash
pvpython OCB_in_PV.py
```
*Output: Generates `.csv` trace data and `.png` visual renders.*

### Step 2: Process Tracing Data (OCB Extraction)
Run the post-processing script to extract the OCB boundaries from the ParaView CSVs.
```bash
python OCB_GM.py
```
*Output: Text files containing MLAT/MLT coordinates of the Open-Closed Boundary.*

### Step 3: Global Toggles & Keogram Generation
Open `main.py` and adjust the **Global Toggles** at the top of the file:
* `X_LINE` (0 or 1): Isolate data at the magnetic reconnection X-line vs. processing the global domain.
* `EMIC_index` (0 or 1): Toggle between Ideal MHD variables and EMIC (Multi-fluid/PIC) variables.
* `SM` (0 or 1): Transform coordinate frames to Solar Magnetic (SM).
* `TANGENTIAL` (0 or 1): Compute tangential force/velocity components across the boundary.

Once paths are configured for your specific event (e.g., `20181020`), run the main pipeline:
```bash
python main.py
```
*Output: Generates Keogram `.eps`/`.png` plots and saves binned data arrays.*

### Step 4: Convection Mapping (Optional)
To generate PyBats ionospheric convection maps with quiver plots, utilize the `convection()` function found in `plot_keogram.py`.

---

## Data Input Requirements
The pipeline expects `.csv` files exported from ParaView or SWMF/BATSRUS `.idl` files for the ionosphere.
* **Ideal MHD**: Requires columns like `x, y, z, bx, by, bz, ux, uy, uz, jx, jy, jz, rho`, and pressure gradients (`gradp0`, etc.).
* **EMIC**: Requires species-specific columns like `rhos0, rhos1, uxs0, uxs1` (electron/ion densities and velocities).
```