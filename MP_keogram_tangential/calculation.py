#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics calculation module.
Computes cross products, tension forces, Alfven velocities, and identifies 
magnetic reconnection X-lines. Prepares data fields for Keogram visualization.
"""

import sys
import numpy as np
from read_data import Data, GSM2SM

def get_MLAT(z, Re):
    """Calculates Magnetic Latitude (MLAT) given Z and radial distance (Re)."""
    # Simple spherical geometry: Latitude = arcsin(Z / Radius)
    return np.arcsin(z / Re) / np.pi * 180

def getMLT(x, y):
    """Calculates Magnetic Local Time from Cartesian X and Y."""
    mlt = np.arctan2(y, x) / np.pi * 180 / 15 + 12
    return np.mod(mlt, 24)

def read_IE_tracing(file, file_used, time, EMIC_index, X_LINE):
    """Reads tracing boundary limits, converts GSM to SM, and adjusts phase shifting."""
    a = np.loadtxt(file, dtype=float, comments='#')
    boundary_mlt = np.zeros((file_used, 2)) * np.nan

    # Split Dawn and Dusk limits, convert frames, and calculate MLT bounds
    dawn_GSM = a[..., 0:3]
    dawn_SM = GSM2SM(dawn_GSM, time, car=1)
    dusk_GSM = a[..., 3:6]
    dusk_SM = GSM2SM(dusk_GSM, time, car=1)
    
    dawn_mlt = getMLT(dawn_SM[..., 0], dawn_SM[..., 1])
    dusk_mlt = getMLT(dusk_SM[..., 0], dusk_SM[..., 1])
    
    boundary_mlt[0:len(dawn_mlt), 0] = dawn_mlt 
    boundary_mlt[0:len(dawn_mlt), 1] = dusk_mlt 

    # Phase shifting alignments depending on the simulation type and X-line focus
    if EMIC_index != 1:
        if X_LINE == 1:
            boundary_mlt = np.roll(boundary_mlt, 2, axis=0)
        elif X_LINE == 0:
            boundary_mlt = np.roll(boundary_mlt, 5, axis=0)
            boundary_mlt[0:5, :] = np.nan
    else:
        boundary_mlt = np.roll(boundary_mlt, 5, axis=0)

    return boundary_mlt

def getXlineBnorm(data_all, zband=20, BnormUezDiff=10, EMIC_index=0):
    """
    Identifies the magnetic reconnection X-line based on B_normal sign changes.
    At a reconnection X-line, the magnetic field component normal to the current 
    sheet (B_n) drops to zero and reverses direction.
    """
    data = data_all.data
    names = data_all.names
    
    ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
    ibx, iby, ibz = names.index("bx"), names.index("by"), names.index("bz")
    inx, iny, inz = names.index("normals0"), names.index("normals1"), names.index("normals2")
    
    results = np.zeros(np.shape(data)) * np.nan
    xline = np.zeros(np.shape(data)) * np.nan
    index = np.zeros(data.shape[0])
    
    # Calculate Normal B component: B_n = B dot Normal_Vector
    Bnorm = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            Bnorm[i, j] = np.dot(data[i, j, [ibx, iby, ibz]], data[i, j, [inx, iny, inz]])
                
    BnormSigns = BnormChangeSigns(Bnorm)
    # Get initial guess for X-line location using electron outflow velocity (Uez ~ 0)
    XlineUez, Uez_index = getXlineUez(data, names, uband=10, EMIC_index=EMIC_index)  

    k = 0
    # Refine the X-line location by looking for the B_normal sign reversal 
    # in the immediate vicinity of the Uez stagnation point.
    for i in range(data.shape[0]):
        if Uez_index[i] == -1:
            index[i] = -1
            continue
            
        temp = np.argwhere(BnormSigns[i, Uez_index[i] : Uez_index[i] + 2 * BnormUezDiff] == 1)[:, 0]
        if len(temp) < 1:
            # Fall back to Uez guess if no clear B_n reversal is found
            index[i] = Uez_index[i]
            XlineBnormindex = Uez_index[i]
            results[i, XlineBnormindex : zband + XlineBnormindex, :] = data[i, XlineBnormindex : zband + XlineBnormindex, :]
            xline[i, XlineBnormindex : 2 + XlineBnormindex, :] = data[i, XlineBnormindex : 2 + XlineBnormindex, :]
            k += 1
        else:
            # Adjust index to the exact point of B_n sign change
            temp1 = np.argmin(np.abs(temp - BnormUezDiff))
            XlineBnormindex = Uez_index[i] - BnormUezDiff + temp[temp1]
            index[i] = XlineBnormindex
            results[i, XlineBnormindex : zband + XlineBnormindex, :] = data[i, XlineBnormindex : zband + XlineBnormindex, :]    
            xline[i, XlineBnormindex : 2 + XlineBnormindex, :] = data[i, XlineBnormindex : 2 + XlineBnormindex, :]  
    
    print(f'There are {k} data points outside the expected range of Uez')
    return Data(results, names), xline

def BnormChangeSigns(Bnorm):
    """
    Flags points where the normal magnetic field component reverses sign.
    Uses a 6-point sliding window to ensure the reversal is a physical trend 
    (+++ followed by ---) rather than local noise.
    """
    index = np.zeros(np.shape(Bnorm)) 
    for i in range(Bnorm.shape[0]):
        for j in range(Bnorm.shape[1] - 5):
            if Bnorm[i, j] > 0 and Bnorm[i, j+1] > 0 and Bnorm[i, j+2] > 0:
                if Bnorm[i, j+3] < 0 and Bnorm[i, j+4] < 0 and Bnorm[i, j+5] < 0:
                    index[i, j] = 1
    return index

def getXlineUez(data, names, uband=2, EMIC_index=0):
    """
    Filters data for regions where U_ez (electron outflow velocity in Z) is near zero.
    The center of an X-line marks a stagnation point where outflows diverge.
    """
    ix, iy, iz = names.index('x'), names.index('y'), names.index('z')
    iuz = names.index('uz') if EMIC_index == 0 else names.index('uzs0')
    
    results = np.zeros(np.shape(data)) * np.nan
    index = np.zeros(data.shape[0]).astype(int)
    
    # Keep data only where velocity is within the near-zero "uband"
    temp = np.argwhere(np.abs(data[:, :, iuz]) < uband)
    results[temp[:, 0], temp[:, 1], :] = data[temp[:, 0], temp[:, 1], :] 
    
    # Restrict to a specific altitude/Z limit to avoid polar cusp confusion
    z_limit = 6 if EMIC_index == 0 else 7
    temp_z = np.argwhere(np.abs(results[:, :, iz]) > z_limit)
    results[temp_z[:, 0], temp_z[:, 1], :] = np.nan
    
    # Record the median index of the stagnation points per column
    for i in range(data.shape[0]):
        valid_x = ~np.isnan(results[i, :, ix])
        if not np.any(valid_x):
            index[i] = -1
        else:
            if EMIC_index == 1:
                index[i] = int(np.argwhere(valid_x)[-1])
            else:
                index[i] = int(np.median(np.argwhere(valid_x)))
                
    return results, index   

def GetCrossProduct(U, B, method=0):
    """
    Computes cross product. 
    Method 0: Standard U x B (used for electric field E = - U x B).
    Method 1: Extracts U_perp (perpendicular velocity to B field).
    """
    Ux, Uy, Uz = U[..., 0], U[..., 1], U[..., 2]
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
    
    if method == 0: # Standard cross product
        UxB_x = (Uy * Bz - Uz * By)
        UxB_y = (Uz * Bx - Ux * Bz) 
        UxB_z = (Ux * By - Uy * Bx) 
    elif method == 1: # Extract Perpendicular component U_perp = U - U_parallel
        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        # U dot B unit vector projection
        udotb = (Ux * Bx + Uy * By + Uz * Bz) / (B_mag**2)
        UxB_x = Ux - udotb * Bx
        UxB_y = Uy - udotb * By
        UxB_z = Uz - udotb * Bz  
        
    UxB_mag = np.sqrt(UxB_x**2 + UxB_y**2 + UxB_z**2)
    UxB = np.zeros((U.shape[0], U.shape[1], 3))
    UxB[..., 0], UxB[..., 1], UxB[..., 2] = UxB_x, UxB_y, UxB_z
    return UxB, UxB_mag

def GetTensionForce(data, names):
    """
    Calculates magnetic tension force mathematically: (B \cdot \nabla) B.
    This force acts to straighten bent magnetic field lines.
    """
    ibx, iby, ibz = names.index("bx"), names.index("by"), names.index("bz")
    # Gradients of B extracted from simulation output
    GradBx, GradBy, GradBz = names.index('gradbx0'), names.index('gradby0'), names.index('gradbz0')
    
    # Dot product of B with the gradient operator applied to each component
    Tensionx = data[..., ibx]*data[..., GradBx] + data[..., iby]*data[..., GradBx+1] + data[..., ibz]*data[..., GradBx+2]
    Tensiony = data[..., ibx]*data[..., GradBy] + data[..., iby]*data[..., GradBy+1] + data[..., ibz]*data[..., GradBy+2]
    Tensionz = data[..., ibx]*data[..., GradBz] + data[..., iby]*data[..., GradBz+1] + data[..., ibz]*data[..., GradBz+2]
         
    F = np.zeros((np.shape(Tensionx)[0], np.shape(Tensionx)[1], 3))
    F[..., 0], F[..., 1], F[..., 2] = Tensionx, Tensiony, Tensionz
    F_mag = np.sqrt(F[..., 0]**2 + F[..., 1]**2 + F[..., 2]**2)
    
    return F, F_mag

def GetAlfven(data, names, xline, EMIC_index=0):
    """
    Calculates the Alfven velocity (V_A) along the guide field of the X-line.
    Math: V_A = B / sqrt(mu_0 * rho). This determines the speed at which 
    magnetic disturbances propagate along the field lines.
    """
    ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
    ibx, iby, ibz = names.index("bx"), names.index("by"), names.index("bz")
    
    if EMIC_index == 0:
        rho = data[..., names.index('rho')]
    else:
        rho = data[..., names.index('rhos1')] + data[..., names.index('rhos0')]
    
    Va = np.zeros((np.shape(data)[0], np.shape(data)[1])) * np.nan
    for i in range(np.shape(data)[0] - 1):
        # Calculate X-line direction vector
        X0 = np.nanmean(xline[i, :, [ix, iy, iz]], axis=1)
        X1 = np.nanmean(xline[i+1, :, [ix, iy, iz]], axis=1)
        xline_dir = X1 - X0
        mag = np.sqrt(xline_dir[0]**2 + xline_dir[1]**2 + xline_dir[2]**2)
        if mag > 0:
            xline_dir = xline_dir / mag
        
        # Project B field onto the X-line direction
        B = data[i, :, [ibx, iby, ibz]]
        B_guide = np.dot(xline_dir, B)
        # 1.66 factor accounts for AMU unit mass conversion used in the plasma code
        Va[i, :] = np.sqrt(B_guide**2 / (4 * np.pi * rho[i, :] * 1.66)) * 100
        
    Va[-1, :] = Va[-2, :]
    return Va

def GetTplotNames_ideal(data_input, tplotnames, SM, xline=None, file=None):
    """
    Extracts, calculates, and bundles the requested tplot variables for ideal MHD.
    Includes JxB forces, tension, perpendicular velocities, and gradients.
    """
    data, names = data_input.data, data_input.names
    ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
    ibx, iby, ibz = names.index("bx"), names.index("by"), names.index("bz")
    iux, iuy, iuz = names.index("ux"), names.index("uy"), names.index("uz")
    ijx, ijy, ijz = names.index("jx"), names.index("jy"), names.index("jz")

    # Calculate derived vector fields
    Uperp, _ = GetCrossProduct(data[..., iux:iux+3], data[..., ibx:ibx+3], method=1)
    E, _ = GetCrossProduct(-1 * data[..., iux:iux+3], data[..., ibx:ibx+3], method=0)
    JxB, _ = GetCrossProduct(data[..., ijx:ijx+3], data[..., ibx:ibx+3], method=0)
    
    F, _ = GetTensionForce(data, names)
    F = F / 6.371 / 4 / np.pi / 10**17 * 10**15  # Normalization constants for plotting scale
    
    results = np.zeros((data.shape[0], data.shape[1], len(tplotnames)))
    
    # Store native variables
    for i, tplotname in enumerate(tplotnames[0:12]):
        results[..., i] = data[..., names.index(tplotname)]
    
    # Normalize pressure gradients
    gradp = -1 * results[..., tplotnames.index('gradp0'):tplotnames.index('gradp0')+3] / 6.371
    results[..., tplotnames.index('gradp0'):tplotnames.index('gradp0')+3] = gradp
        
    results[..., tplotnames.index('gradpb0'):tplotnames.index('gradpb0')+3] = \
        -1 * results[..., tplotnames.index('gradpb0'):tplotnames.index('gradpb0')+3] / 6.371 / 100

    # Store calculated vector fields
    results[..., tplotnames.index('uperpx'):tplotnames.index('uperpx')+3] = Uperp
    results[..., tplotnames.index('Ex'):tplotnames.index('Ex')+3] = E
    results[..., tplotnames.index('JxBx'):tplotnames.index('JxBx')+3] = JxB
    results[..., tplotnames.index('Tensionx'):tplotnames.index('Tensionx')+3] = F

    # Calculate force balances
    results[..., tplotnames.index('JxBcalx'):tplotnames.index('JxBcalx')+3] = \
        F + results[..., tplotnames.index('gradpb0'):tplotnames.index('gradpb0')+3] 
        
    results[..., tplotnames.index('Ftotalx'):tplotnames.index('Ftotalx')+3] = JxB + gradp
    
    if 'Va' in names:
        results[..., tplotnames.index('Va')] = GetAlfven(data, names, xline, EMIC_index=0)
        
    # Calculate Poleward Velocity (u_pole) in the SM reference frame
    if 'upole' in tplotnames:
        time = '2018-10-20T21:' + file[-6:-4]
        X_SM = data[..., ix:ix+3]
        uperpx_SM = GSM2SM(Uperp, time, car=1) if SM == 0 else Uperp
            
        Re = np.sqrt(X_SM[..., 0]**2 + X_SM[..., 1]**2 + X_SM[..., 2]**2)
        mlat = get_MLAT(X_SM[..., 2], Re)
        
        # Project velocity onto horizontal and poleward directions
        u_hor = -(uperpx_SM[..., 0]*X_SM[..., 0] + uperpx_SM[..., 1]*X_SM[..., 1]) / Re / np.cos(mlat / 180 * np.pi)
        u_pole = uperpx_SM[..., 2]*np.cos(mlat / 180 * np.pi) + u_hor * np.sin(mlat / 180 * np.pi)
        results[..., tplotnames.index('upole')] = u_pole

    if 'Ftotalcalx' in tplotnames:
        results[..., tplotnames.index('Ftotalcalx'):tplotnames.index('Ftotalcalx')+3] = \
            F + results[..., tplotnames.index('gradpb0'):tplotnames.index('gradpb0')+3] + gradp
            
    return results