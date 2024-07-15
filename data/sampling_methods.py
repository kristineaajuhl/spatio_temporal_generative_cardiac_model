# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:57:11 2023

@author: kajul
"""

import numpy as np
import os
import random
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import pymeshlab
import plyfile
#import matplotlib.pyplot as plt
#import nibabel as nib
import time

#%% METHODS
def point_to_surf_distance_wSign(points,pd):
   implicitPolyDataDistance = vtk.vtkImplicitPolyDataDistance()
   implicitPolyDataDistance.SetInput(pd)
   
   all_sdf = np.zeros(points.shape[0]).astype(np.float32)
   for i, point in enumerate(points):
       closest_point = [0.0,0.0,0.0]
       signedDistance =  implicitPolyDataDistance.EvaluateFunctionAndGetClosestPoint( point, closest_point )
       all_sdf[i] = signedDistance
       
   sdf_signcorrected = check_sign(points,all_sdf,pd)
   return sdf_signcorrected

def check_sign(points,all_sdf,pd):
    # pd to stencil
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(pd)
    pol2stenc.SetOutputOrigin([-0.5, -0.5, -0.5])
    pol2stenc.SetOutputSpacing([0.01, 0.01, 0.01])
    pol2stenc.SetOutputWholeExtent((0, 99, 0, 99, 0, 99))
    pol2stenc.Update()

    spacing = [0.01,]*3  # Set your desired spacing
    origin = [-0.5, -0.5, -0.5]       # Set your desired origin
    extent = 100

    image_data = vtk.vtkImageData()
    image_data.SetSpacing(spacing)
    image_data.SetOrigin(origin)
    image_data.SetExtent(0, extent, 0, extent, 0, extent)  # Set your desired extent
    scalar_values = vtk.vtkDoubleArray()
    scalar_values.SetNumberOfComponents(1)
    scalar_values.SetNumberOfTuples(image_data.GetNumberOfPoints())
    scalar_values.Fill(1.0)
    image_data.GetPointData().SetScalars(scalar_values)


    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(image_data)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    mask = imgstenc.GetOutput()
    # np_mask = np.reshape(vtk_to_numpy(mask.GetPointData().GetScalars()),(extent+1,)*3)
    # #image_coord = np.floor((points+0.5)*(extent+1)).astype(int)
    # #samples = np_mask[image_coord[:,0],image_coord[:,1],image_coord[:,2]]

    # # Create a NIfTI image
    # nifti_image = nib.Nifti1Image(np_mask, affine=np.eye(4))  # Use identity matrix as the affine

    # # Save the NIfTI image to a file
    # nib.save(nifti_image, "F:/DATA/LALAA/mask/" + str(time.time())+'.nii.gz')

    
    ### VTK probefilter
    # Sample points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.array(points)))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    # Create vtkProbeFilter
    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetInputData(polydata)
    probe_filter.SetSourceData(mask)
    probe_filter. SetCategoricalData(True)
    probe_filter.Update()

    # Get the probed values
    probed_values = probe_filter.GetOutput().GetPointData().GetScalars()
    np_probed = vtk_to_numpy(probed_values)
    sdf_inside = (all_sdf<0).astype(int)

    disagree = np_probed != sdf_inside
    large_sdf = abs(all_sdf)>spacing[0]
    change_sign = (disagree.astype(int) + large_sdf.astype(int))==2
    all_sdf[change_sign] = -all_sdf[change_sign]
    #print("Number of disagreements: ", np.sum(disagree))
    #print("Minimum disagreeing SDF: ", np.min(abs(all_sdf[disagree]))) 
    #print("maximum disagreeing SDF: ", np.max(abs(all_sdf[disagree])))
    #print("Number of samples where sign is changed: ", np.sum(change_sign))

    return all_sdf



def volatile_shapediameter_sampling_nonormal(args):
    norm_file = args[0]
    sample_sigma = args[1]
    n_samples = args[2]
    volatile_factor = args[3]
    volatile_threshold = args[4]
    #volatile_threshold_in, volatile_threshold_out  = args[4]
    shapediameter_dir = args[5]
    
    if not os.path.exists(shapediameter_dir):
        os.mkdir(shapediameter_dir)
    
    shape_diameter_file_out = shapediameter_dir + '/shape_diameter_out.vtk'
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(norm_file)
    reader.Update()
    surf_out = reader.GetOutput()   
   
    if not os.path.exists(shape_diameter_file_out):
        # Save temprary PLY file to use with meshlab
        temp_input_file = os.path.join(shapediameter_dir,str(time.time())+"_temp_inputfile.ply")#shapediameter_dir + '/temp_inputfile.ply'
        temp_output_file= os.path.join(shapediameter_dir,str(time.time())+"_temp_outputfile.ply")#shapediameter_dir + '/temp_outputfile.ply'
        writer = vtk.vtkPLYWriter()
        writer.SetInputData(surf_out)
        writer.SetFileName(temp_input_file)
        writer.Write()        
        
        # Calculate shape diameter function
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_input_file)
        #ms.shape_diameter_function(onprimitive='On Faces', coneangle=45)
        ms.compute_scalar_by_shape_diameter_function_per_vertex(onprimitive='On Faces', coneangle=45)
        ms.save_current_mesh(temp_output_file, binary=False, save_vertex_color = False, save_face_color=False)
        
        # Save shape diameter information on polydata cell scalars
        plydata = plyfile.PlyData.read(temp_output_file)
        shape_diameter = plydata['face'].data['quality']
        
        surf_out.GetCellData().SetScalars(numpy_to_vtk(shape_diameter))

        # Smooth cell scalars using vtkWindowedSincPolyDataFilter
        #smooth_filter = vtk.vtkWindowedSincPolyDataFilter()
        #smooth_filter.SetInputData(surf_out)
        #smooth_filter.SetNumberOfIterations(20)  # Adjust the number of iterations as needed
        #smooth_filter.BoundarySmoothingOff()
        #smooth_filter.FeatureEdgeSmoothingOff()
        #smooth_filter.SetPassBand(0.1)  # Adjust pass band value as needed
        #smooth_filter.Update()
        
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(shape_diameter_file_out)
        writer.SetInputData(surf_out)
        writer.Write()
        
        os.remove(temp_input_file)
        os.remove(temp_output_file)
    else: 
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(shape_diameter_file_out)
        reader.Update()
        surf_out = reader.GetOutput()
        
    # Do the same with flipped normals
    shape_diameter_file_in = shapediameter_dir + '/shape_diameter_in.vtk'
    reverseSense = vtk.vtkReverseSense()
    reverseSense.SetInputData(surf_out)
    reverseSense.ReverseNormalsOn()
    reverseSense.Update()
    surf_in = reverseSense.GetOutput()
    
    if not os.path.exists(shape_diameter_file_in):
        # Save temprary PLY file to use with meshlab
        temp_input_file = os.path.join(shapediameter_dir,str(time.time())+'in_temp_inputfile.ply') #shapediameter_dir + '/in_temp_inputfile.ply'
        temp_output_file= os.path.join(shapediameter_dir,str(time.time())+'in_temp_outputfile.ply') #shapediameter_dir + '/in_temp_outputfile.ply'
        writer = vtk.vtkPLYWriter()
        writer.SetInputData(surf_in)
        writer.SetFileName(temp_input_file)
        writer.Write()        
        
        # Calculate shape diameter function
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(temp_input_file)
        ms.compute_scalar_by_shape_diameter_function_per_vertex(onprimitive='On Faces',coneangle=45)
        ms.save_current_mesh(temp_output_file, binary=False, save_vertex_color = False, save_face_color=False)
        
        # Save shape diameter information on polydata cell scalars
        plydata = plyfile.PlyData.read(temp_output_file)
        shape_diameter = plydata['face'].data['quality']
        
        surf_in.GetCellData().SetScalars(numpy_to_vtk(shape_diameter))
        
        # Smooth cell scalars using vtkWindowedSincPolyDataFilter
        #smooth_filter = vtk.vtkWindowedSincPolyDataFilter()
        #smooth_filter.SetInputData(surf_in)
        #smooth_filter.SetNumberOfIterations(2000)  # Adjust the number of iterations as needed
        #smooth_filter.BoundarySmoothingOff()
        #smooth_filter.FeatureEdgeSmoothingOff()
        #smooth_filter.SetPassBand(0.1)  # Adjust pass band value as needed
        #smooth_filter.Update()
        
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(shape_diameter_file_in)
        writer.SetInputData(surf_in)
        writer.Write()
        
        os.remove(temp_input_file)
        os.remove(temp_output_file)
    else: 
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(shape_diameter_file_in)
        reader.Update()
        surf_in = reader.GetOutput()
     
    boundary_points_1 = []
    boundary_points_2 = []
        
    normals_in = surf_in.GetPointData().GetNormals()
    n_cells = surf_in.GetNumberOfCells()
    sd_vec = vtk_to_numpy(surf_in.GetCellData().GetScalars())
    sd_vec_out = vtk_to_numpy(surf_out.GetCellData().GetScalars())
    volatile_sampling_vec = np.repeat(np.arange(0,n_cells),repeats=(sd_vec<volatile_threshold)*volatile_factor+1)
    for i in range(n_samples):
        cellId = np.random.choice(volatile_sampling_vec,1)[0]
        triangle = surf_in.GetCell(cellId).GetPoints()
        #sd_in = surf_in.GetCellData().GetScalars().GetTuple(cellId)[0]
        sd_in = np.min([sd_vec[cellId],sample_sigma*2])
        sd_out = np.min([sd_vec_out[cellId],sample_sigma*2])
        
        p1_normal = np.array(normals_in.GetTuple(surf_in.GetCell(cellId).GetPointIds().GetId(0)))
        p2_normal = np.array(normals_in.GetTuple(surf_in.GetCell(cellId).GetPointIds().GetId(1)))
        p3_normal = np.array(normals_in.GetTuple(surf_in.GetCell(cellId).GetPointIds().GetId(2)))
        
        r = [random.random() for i in range(0,3)]
        bary = [i/sum(r) for i in r]
        
        point_x = bary[0] * triangle.GetPoint(0)[0] + bary[1] * triangle.GetPoint(1)[0] + bary[2] * triangle.GetPoint(2)[0]
        point_y = bary[0] * triangle.GetPoint(0)[1] + bary[1] * triangle.GetPoint(1)[1] + bary[2] * triangle.GetPoint(2)[1]
        point_z = bary[0] * triangle.GetPoint(0)[2] + bary[1] * triangle.GetPoint(1)[2] + bary[2] * triangle.GetPoint(2)[2]
        point = [point_x, point_y, point_z]

        noise = np.random.normal(0.0,sd_in/2,3)
        boundary_points_1.append(point + noise)
        noise = np.random.normal(0.0,sd_in,3)
        boundary_points_1.append(point + noise)

        noise = np.random.normal(0.0,sd_out/2,3)
        boundary_points_2.append(point + noise)
        noise = np.random.normal(0.0,sd_out,3)
        boundary_points_2.append(point + noise)
    
       
    occupancies_1 = point_to_surf_distance_wSign(np.array(boundary_points_1),surf_out)
    occupancies_2 = point_to_surf_distance_wSign(np.array(boundary_points_2),surf_out)
    
    return boundary_points_1, boundary_points_2, occupancies_1, occupancies_2