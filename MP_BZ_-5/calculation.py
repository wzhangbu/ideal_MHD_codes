#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics calculation module.
Computes cross products, tension forces, Alfven velocities, and identifies 
magnetic reconnection X-lines. Prepares data fields for Keogram visualization.
"""

import sys
import os
import numpy as np
from spacepy import coordinates as coord
from spacepy.time import Ticktock
from read_data import Data

def GSM2SM(X, time, car=1):
    """Converts 3D vectors from GSM to SM coordinate frames."""
    X_temp = X.reshape(-1, 3)
    X_GSM = coord.Coords(X_temp, 'GSM', 'car')
    times = np.repeat(time, len(X_temp[:, 0]), axis=0)
    X_GSM.ticks = Ticktock(times)
    
    if car == 1:
        X_out = X_GSM.convert('SM', 'car')
    else:
        X_out = X_GSM.convert('SM', 'sph')
    
    return X_out.data.reshape(X.shape)

def get_MLAT(z, Re):
    """Calculates Magnetic Latitude (MLAT) given Z and radial distance (Re)."""
    return np.arcsin(z / Re) / np.pi * 180

def getMLT(x, y):
    """Calculates Magnetic Local Time from Cartesian X and Y coordinates."""
    mlt = np.arctan2(y, x) / np.pi * 180 / 15 + 12
    return np.mod(mlt, 24)

def read_IE_tracing(file, file_used, time, EMIC_index, X_LINE):
    """Reads OCB tracing boundary limits, converts GSM to SM, and adjusts phase shifting."""
    a = np.loadtxt(file, dtype=float, comments='#')
    boundary_mlt = np.zeros((file_used, 2)) * np.nan

    # Split into Dawn and Dusk limits
    dawn_GSM = a[..., 0:3]
    dawn_SM = GSM2SM(dawn_GSM, time, car=1)
    
    dusk_GSM = a[..., 3:6]
    dusk_SM = GSM2SM(dusk_GSM, time, car=1)
    
    dawn_mlt = getMLT(dawn_SM[..., 0], dawn_SM[..., 1])
    dusk_mlt = getMLT(dusk_SM[..., 0], dusk_SM[..., 1])
    
    boundary_mlt[0:len(dawn_mlt), 0] = dawn_mlt 
    boundary_mlt[0:len(dawn_mlt), 1] = dusk_mlt 

    # Apply phase shifts to align data based on the specific simulation mode
    if EMIC_index != 1:
        if X_LINE == 1:
            boundary_mlt = np.roll(boundary_mlt, 2, axis=0)
        if X_LINE == 0:
            boundary_mlt = np.roll(boundary_mlt, 5, axis=0)
            boundary_mlt[0:5, :] = np.nan
    elif EMIC_index == 1:
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
    for i in range(data.shape[0]):
        if Uez_index[i] == -1:
            index[i] = -1
            continue
            
        # Search for sign flip within a bounding window
        temp = np.argwhere(BnormSigns[i, Uez_index[i] : Uez_index[i] + 2*BnormUezDiff] == 1)[:, 0]
        if len(temp) < 1:
            # Fall back to Uez guess if no clear B_n reversal is found
            index[i] = Uez_index[i]
            XlineBnormindex = Uez_index[i]
            results[i, XlineBnormindex: zband + XlineBnormindex, :] = data[i, XlineBnormindex: zband + XlineBnormindex, :]
            xline[i, XlineBnormindex: 2 + XlineBnormindex, :] = data[i, XlineBnormindex: 2 + XlineBnormindex, :]
            k += 1
        else:
            # Adjust index to the exact point of B_n sign change
            temp1 = np.argmin(np.abs(temp - BnormUezDiff))
            XlineBnormindex = Uez_index[i] - BnormUezDiff + temp[temp1]
            index[i] = XlineBnormindex
            results[i, XlineBnormindex: zband + XlineBnormindex, :] = data[i, XlineBnormindex:zband + XlineBnormindex, :]    
            xline[i, XlineBnormindex: 2 + XlineBnormindex, :] = data[i, XlineBnormindex:2 + XlineBnormindex, :]  
    
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
    The center of an X-line marks a stagnation point where vertical outflows diverge.
    """
    ix, iy, iz = names.index('x'), names.index('y'), names.index('z')
    iuz = names.index('uz') if EMIC_index == 0 else names.index('uzs0')
    
    results = np.zeros(np.shape(data)) * np.nan
    index = np.zeros(data.shape[0]).astype(int)
    
    # Keep data only where velocity is within the near-zero "uband"
    temp = np.argwhere((np.abs(data[:, :, iuz]) < uband))
    results[temp[:, 0], temp[:, 1], :] = data[temp[:, 0], temp[:, 1], :] 
    
    # Restrict to a specific altitude/Z limit to avoid polar cusp confusion
    z_limit = 6 if EMIC_index == 0 else 7
    temp_z = np.argwhere(np.abs(results[:, :, iz]) > z_limit)
    results[temp_z[:, 0], temp_z[:, 1], :] = np.nan
    
    # Record the index of the stagnation point per column
    for i in range(data.shape[0]):
        temp = ~np.isnan(results[i, :, ix])
        if len(results[i, temp, ix]) < 1:
            index[i] = -1
        else:
            index[i] = int(np.median(np.argwhere(~np.isnan(results[i, :, ix]))))
            if EMIC_index == 1:
                index[i] = int(np.argwhere(~np.isnan(results[i, :, ix]))[-1])
                
    return results, index   

def GetCrossProduct(U, B, method=0):
    """
    Computes cross product. 
    Method 0: Standard U x B (used for electric field E = - U x B).
    Method 1: Extracts U_perp (perpendicular velocity to B field).
    """
    Ux, Uy, Uz = U[..., 0], U[..., 1], U[..., 2]
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]
    
    if method == 0:
        # Standard Cross Product Vector Math
        UxB_x = (Uy * Bz - Uz * By)
        UxB_y = (Uz * Bx - Ux * Bz) 
        UxB_z = (Ux * By - Uy * Bx) 
    elif method == 1:
        # U_perp calculation: Subtract the field-aligned velocity from total velocity
        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        udotb = (Ux * Bx + Uy * By + Uz * Bz) / (B_mag**2)
        
        UxB_x = Ux - udotb * Bx
        UxB_y = Uy - udotb * By
        UxB_z = Uz - udotb * Bz  
        
    UxB_mag = np.sqrt(UxB_x**2 + UxB_y**2 + UxB_z**2)
    UxB = np.zeros((U.shape[0], U.shape[1], 3))
    UxB[..., 0] = UxB_x; UxB[..., 1] = UxB_y; UxB[..., 2] = UxB_z
    return UxB, UxB_mag

def GetTensionForce(data, names):
    """
    Calculates magnetic tension force mathematically: (B \cdot \nabla) B.
    This force acts to straighten bent magnetic field lines.
    """
    ibx, iby, ibz = names.index("bx"), names.index("by"), names.index("bz")
    GradBx, GradBy, GradBz = names.index('gradbx0'), names.index('gradby0'), names.index('gradbz0')
    
    # Dot product of B with the gradient operator applied to each component
    Tensionx = data[..., ibx] * data[..., GradBx] + data[..., iby] * data[..., GradBx + 1] + data[..., ibz] * data[..., GradBx + 2]
    Tensiony = data[..., ibx] * data[..., GradBy] + data[..., iby] * data[..., GradBy + 1] + data[..., ibz] * data[..., GradBy + 2]
    Tensionz = data[..., ibx] * data[..., GradBz] + data[..., iby] * data[..., GradBz + 1] + data[..., ibz] * data[..., GradBz + 2]
         
    F = np.zeros((np.shape(Tensionx)[0], np.shape(Tensionx)[1], 3))
    F[..., 0] = Tensionx; F[..., 1] = Tensiony; F[..., 2] = Tensionz
    F_mag = np.sqrt(F[..., 0]**2 + F[..., 1]**2 + F[..., 2]**2)
    
    return F, F_mag

def GetAlfven(data, names, xline, EMIC_index=0):
    """
    Calculates the Alfven velocity (V_A) along the guide field of the X-line.
    Math: V_A = B / sqrt(mu_0 * rho).
    """
    ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
    ibx, iby, ibz = names.index("bx"), names.index("by"), names.index("bz")
    
    if EMIC_index == 0:
        rho = data[..., names.index('rho')]
    elif EMIC_index == 1:
        rho = data[..., names.index('rhos1')] + data[..., names.index('rhos0')]
    
    Va = np.zeros((np.shape(data)[0], np.shape(data)[1])) * np.nan
    for i in range(np.shape(data)[0]-1):
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
        Va[i, :] = np.sqrt(B_guide**2 / 4 / np.pi / rho[i, :] / 1.66) * 100
        
    Va[-1, :] = Va[-2, :]
    return Va

def GetTangential(vec, norm):
    """Extracts the tangential components of a vector field relative to a boundary normal."""
    if np.shape(vec)[2] % 3 != 0:
        sys.exit('Size error, please check vector dimensions')
    else:
        for i in range(int(np.shape(vec)[2] / 3)):
            temp = vec[..., 3*i : 3*i + 3]
            udotnorm = temp[..., 0] * norm[..., 0] + temp[..., 1] * norm[..., 1] + temp[..., 2] * norm[..., 2]

            for j in range(3):
                temp[..., j] = temp[..., j] - udotnorm * norm[..., j]
            vec[..., 3*i : 3*i+3] = temp
    return vec

def GetTplotNames_ideal_tang(data_input, tplotnames, SM, xline=None, file=None, tangential=None):
    """Extracts, formats, and returns the requested tplot variables for ideal MHD."""
    data, names = data_input.data, data_input.names
    ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
    ibx, iby, ibz = names.index("bx"), names.index("by"), names.index("bz")
    iux, iuy, iuz = names.index("ux"), names.index("uy"), names.index("uz")
    ijx, ijy, ijz = names.index("jx"), names.index("jy"), names.index("jz")
    inormx = names.index('normals0')
    
    # Calculate derived vector fields
    U = data[..., iux:iux + 3]
    Uperp, _ = GetCrossProduct(data[..., iux:iux+3], data[..., ibx:ibx+3], method=1)
    E, _ = GetCrossProduct(-1*data[..., iux:iux+3], data[..., ibx:ibx+3], method=0)
    JxB, _ = GetCrossProduct(data[..., ijx:ijx+3], data[..., ibx:ibx+3], method=0)
    
    F, _ = GetTensionForce(data, names)
    F = F / 6.371 / 4 / np.pi / 10**17 * 10**15  # Normalization constants for plotting
    
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
        
    if 'Ftotalcalx' in tplotnames:
        results[..., tplotnames.index('Ftotalcalx'):tplotnames.index('Ftotalcalx')+3] = \
            F + results[..., tplotnames.index('gradpb0'):tplotnames.index('gradpb0')+3] + gradp

    # Compute tangential specific components if toggled
    if tangential == 1:
        X_SM = data[..., ix:ix+3]
        parameters = ['upole', 'JxB_pole', 'gradp_pole', 'Ftotal_pole',
                      'Ftention_pole', 'Gradpb_pole', 'Ftotal_tangx',
                      'Ftotal_tangy', 'Ftotal_tangz', 'U_perp_tangx',
                      'U_perp_tangy', 'U_perp_tangz']
        time = '2018-10-20T21:' + file[-6:-4]
        
        for para in parameters:
            # Map the variable to be processed
            if para == 'upole': var = results[..., tplotnames.index('uperpx'):tplotnames.index('uperpx')+3]
            elif para == 'JxB_pole': var = results[..., tplotnames.index('JxBx'):tplotnames.index('JxBx')+3]
            elif para == 'gradp_pole': var = results[..., tplotnames.index('gradp0'):tplotnames.index('gradp0')+3]                
            elif para == 'Ftotal_pole': var = results[..., tplotnames.index('Ftotalx'):tplotnames.index('Ftotalx')+3]
            elif para == 'Ftention_pole': var = results[..., tplotnames.index('Tensionx'):tplotnames.index('Tensionx')+3]
            elif para == 'Gradpb_pole': var = results[..., tplotnames.index('gradpb0'):tplotnames.index('gradpb0')+3]
            elif para in ['Ftotal_tangx', 'Ftotal_tangy', 'Ftotal_tangz']:
                var = results[..., tplotnames.index('Ftotalx'):tplotnames.index('Ftotalx')+3]
            elif para in ['U_perp_tangx', 'U_perp_tangy', 'U_perp_tangz']:
                var = results[..., tplotnames.index('uperpx'):tplotnames.index('uperpx')+3]
            
            # Convert to tangential and SM framework
            var = GetTangential(var, data[..., inormx:inormx+3])
            var_SM = GSM2SM(var, time, car=1) if SM == 0 else var
            
            # Store isolated cartesian tangential vectors
            if para == 'Ftotal_tangx': results[..., tplotnames.index(para)] = var_SM[..., 0]; continue
            if para == 'Ftotal_tangy': results[..., tplotnames.index(para)] = var_SM[..., 1]; continue
            if para == 'Ftotal_tangz': results[..., tplotnames.index(para)] = var_SM[..., 2]; continue
            if para == 'U_perp_tangx': results[..., tplotnames.index(para)] = var_SM[..., 0]; continue
            if para == 'U_perp_tangy': results[..., tplotnames.index(para)] = var_SM[..., 1]; continue
            if para == 'U_perp_tangz': results[..., tplotnames.index(para)] = var_SM[..., 2]; continue
            
            # Compute Poleward components dynamically
            Re = np.sqrt(X_SM[..., 0]**2 + X_SM[..., 1]**2 + X_SM[..., 2]**2)
            mlat = get_MLAT(X_SM[..., 2], Re) 
            var_hor = -(var_SM[..., 0] * X_SM[..., 0] + var_SM[..., 1] * X_SM[..., 1]) / Re / np.cos(mlat / 180 * np.pi)
            var_pole = var_SM[..., 2] * np.cos(mlat / 180 * np.pi) + var_hor * np.sin(mlat / 180 * np.pi)
            
            results[..., tplotnames.index(para)] = var_pole

    # Standard Poleward Velocity Calculation
    if 'upole' in tplotnames and tangential == 0:
        time = '2018-10-20T21:' + file[-6:-4]
        X_SM = data[..., ix:ix+3]
        
        if tangential == 1:
            Uperp = GetTangential(Uperp, data[..., inormx:inormx+3])
            
        uperpx_SM = GSM2SM(Uperp, time, car=1) if SM == 0 else Uperp
        
        Re = np.sqrt(X_SM[..., 0]**2 + X_SM[..., 1]**2 + X_SM[..., 2]**2)
        mlat = get_MLAT(X_SM[..., 2], Re)
        u_hor = -(uperpx_SM[..., 0] * X_SM[..., 0] + uperpx_SM[..., 1] * X_SM[..., 1]) / Re / np.cos(mlat / 180 * np.pi)
        u_pole = uperpx_SM[..., 2] * np.cos(mlat / 180 * np.pi) + u_hor * np.sin(mlat / 180 * np.pi)
        results[..., tplotnames.index('upole')] = u_pole

    return results

def GetTplotNames_EMIC(data_input, tplotnames, xline=None):
    """Extracts and formats the requested tplot variables specific to EMIC (Multi-fluid/PIC) simulations."""
    data, names = data_input.data, data_input.names
    ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
    ibx, iby, ibz = names.index("bx"), names.index("by"), names.index("bz")
    ijx, ijy, ijz = names.index("jx"), names.index("jy"), names.index("jz")
    
    iuex, iuey, iuez = names.index("uxs0"), names.index("uys0"), names.index("uzs0")
    iuix, iuiy, iuiz = names.index("uxs1"), names.index("uys1"), names.index("uzs1")
    
    Ueperp, _ = GetCrossProduct(data[..., iuex:iuex+3], data[..., ibx:ibx+3], method=1)
    Uiperp, _ = GetCrossProduct(data[..., iuix:iuix+3], data[..., ibx:ibx+3], method=1)
    JxB, _ = GetCrossProduct(data[..., ijx:ijx+3], data[..., ibx:ibx+3], method=0)
    
    F, _ = GetTensionForce(data, names)
    F = F / 6.371 / 4 / np.pi / 10**17 * 10**15
    
    results = np.zeros((F.shape[0], F.shape[1], len(tplotnames)))
    for i, tplotname in enumerate(tplotnames[0:18]):
        results[..., i] = data[..., names.index(tplotname)]  
    
    gradp = -1 * results[..., tplotnames.index('gradp0'):tplotnames.index('gradp0')+3] / 6.371
    results[..., tplotnames.index('gradp0'):tplotnames.index('gradp0')+3] = gradp
        
    results[..., tplotnames.index('gradpb0'):tplotnames.index('gradpb0')+3] = \
        -1 * results[..., tplotnames.index('gradpb0'):tplotnames.index('gradpb0')+3] / 6.371 / 100

    results[..., tplotnames.index('ueperpx'):tplotnames.index('ueperpx')+3] = Ueperp
    results[..., tplotnames.index('uiperpx'):tplotnames.index('uiperpx')+3] = Uiperp
    results[..., tplotnames.index('JxBx'):tplotnames.index('JxBx')+3] = JxB
    results[..., tplotnames.index('Tensionx'):tplotnames.index('Tensionx')+3] = F

    results[..., tplotnames.index('JxBcalx'):tplotnames.index('JxBcalx')+3] = \
        F + results[..., tplotnames.index('gradpb0'):tplotnames.index('gradpb0')+3] 
        
    results[..., tplotnames.index('Ftotalx'):tplotnames.index('Ftotalx')+3] = JxB + gradp
    
    return results