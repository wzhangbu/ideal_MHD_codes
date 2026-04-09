# Trace generated using paraview version 5.11.2
import glob
import os
from paraview.simple import *

# Disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# --- Config ---
event_path = '/Users/weizhang/Desktop/research/EMIC/20181020/' 
run_list = 'run81_epic/' 
data_list = os.path.join(event_path, run_list, 'GM/')
Fig_dir = os.path.join(event_path, run_list, 'keogram_MLAT/')

os.chdir(data_list) 

def file_names(file_dir):
    """Filters directories to find event folders."""
    File_name = [files for files in os.listdir(file_dir) if os.path.isdir(files)]
    File_name.sort()
    return File_name

def ResetSession():
    """Wipes the active ParaView proxy manager to prevent memory/state leaks."""
    pxm = servermanager.ProxyManager()
    pxm.UnRegisterProxies()
    del pxm
    Disconnect()
    Connect()

files = file_names(data_list)
print(f'There are {len(files)} files in total.')

for i, file in enumerate(files):
    FileDir = os.path.join(data_list, files[i])
    FileName = files[i][:-4] + '.vtm'
    print(f"Processing: {FileName}")
    
    ResetSession()
    
    # Load MultiBlock Data
    a3d_reader = XMLMultiBlockDataReader(registrationName=FileName, FileName=[os.path.join(FileDir, FileName)])
    a3d_reader.PointArrayStatus = ['Rho', 'U_x', 'U_y', 'U_z', 'B_x', 'B_y', 'B_z', 'P', 'pe', 'J_x', 'J_y', 'J_z']
    a3d_reader.TimeArray = 'None'
    UpdatePipeline(time=0.0, proxy=a3d_reader)

    # Extract Cells (Box constraint)
    extractCells = ExtractCellsByRegion(registrationName='ExtractCellsByRegion1', Input=a3d_reader)
    extractCells.IntersectWith = 'Box'
    extractCells.IntersectWith.Position = [-80.0, -30.0, -30.0]
    extractCells.IntersectWith.Length = [100.0, 60.0, 60.0]
    UpdatePipeline(time=0.0, proxy=extractCells)

    # Calculator: B-Vector
    calc_bvec = Calculator(registrationName='Calculator1', Input=extractCells)
    calc_bvec.ResultArrayName = 'B_vec'
    calc_bvec.Function = 'B_x * iHat + B_y * jHat + B_z * kHat'
    UpdatePipeline(time=0.0, proxy=calc_bvec)

    # Calculator: Radius (Re)
    calc_re = Calculator(registrationName='Calculator2', Input=calc_bvec)
    calc_re.ResultArrayName = 'Re'
    calc_re.Function = 'sqrt(coordsX^2 + coordsY^2 +coordsZ^2 )'
    UpdatePipeline(time=0.0, proxy=calc_re)

    # Contour by Radius = 3
    contour1 = Contour(registrationName='Contour1', Input=calc_re)
    contour1.ContourBy = ['POINTS', 'Re']
    contour1.Isosurfaces = [3.0]
    contour1.PointMergeMethod = 'Uniform Binning'
    UpdatePipeline(time=0.0, proxy=contour1)

    # Seed Sphere for Stream Tracer
    sphere1 = Sphere(registrationName='Sphere1')
    sphere1.Radius = 3.0
    sphere1.ThetaResolution = 73
    sphere1.PhiResolution = 41
    sphere1.StartPhi = 1.0
    sphere1.EndPhi = 21.0
    UpdatePipeline(time=0.0, proxy=sphere1)

    calc3 = Calculator(registrationName='Calculator3', Input=sphere1)
    calc3.ResultArrayName = 'Re'
    calc3.Function = 'sqrt(coordsX^2 + coordsY^2 +coordsZ^2 )'
    
    calc4 = Calculator(registrationName='Calculator4', Input=calc3)
    calc4.ResultArrayName = 'Lat'
    calc4.Function = 'asin( coordsZ / Re) * 180 / 3.14159'
    latLUT = GetColorTransferFunction('Lat')

    # Stream Tracer Custom Source (Field lines)
    streamTracer = StreamTracerWithCustomSource(registrationName='StreamTracer1', Input=calc_re, SeedSource=sphere1)
    streamTracer.Vectors = ['POINTS', 'B_vec']
    streamTracer.IntegrationDirection = 'BACKWARD'
    streamTracer.MaximumStreamlineLength = 300.0
    UpdatePipeline(time=0.0, proxy=streamTracer)

    # Project to Points
    cell2point = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=streamTracer)
    cell2point.CellDataArraytoprocess = ['ReasonForTermination', 'SeedIds']
    UpdatePipeline(time=0.0, proxy=cell2point)

    SetActiveSource(streamTracer)
    extractEnclosed = ExtractEnclosedPoints(registrationName='ExtractEnclosedPoints1', Input=cell2point, Surface=sphere1)
    UpdatePipeline(time=0.0, proxy=extractEnclosed)

    # --- Rendering Setup ---
    renderView1 = GetActiveViewOrCreate('RenderView')
    contour2 = Contour(registrationName='Contour2', Input=calc4)
    contour2.ContourBy = ['POINTS', 'Lat']
    contour2.PointMergeMethod = 'Uniform Binning'
    contour2.Isosurfaces = [70.0, 75.0, 80.0, 85.0]

    contour2Display = Show(contour2, renderView1, 'GeometryRepresentation')
    contour2Display.LineWidth = 5.0
    latLUT.RescaleTransferFunction(50.0, 100.0)

    SetActiveSource(extractEnclosed)
    extractDisplay = Show(extractEnclosed, renderView1, 'GeometryRepresentation')
    extractDisplay.Representation = 'Point Gaussian'
    extractDisplay.ColorArrayName = ['POINTS', 'ReasonForTermination']

    # Camera Defaults
    renderView1.CameraPosition = [0.0019, 0.0, 11.50]
    renderView1.CameraFocalPoint = [0.0019, 0.0, 2.90]
    renderView1.CameraViewUp = [1.0, 0.0, 0.0]
    renderView1.CameraParallelScale = 1.0

    # Output Data
    SaveScreenshot(os.path.join(Fig_dir, f"{i + 25}.png"), renderView1, ImageResolution=[2798, 1082])
    SaveData(os.path.join(Fig_dir, f"{i + 25}.csv"), proxy=extractEnclosed, ChooseArraysToWrite=1,
             PointDataArrays=['B_x', 'B_y', 'B_z', 'ReasonForTermination'])