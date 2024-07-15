# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:21:44 2023

@author: kajul
"""

import numpy as np
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from multiprocessing import Pool
from itertools import product
import tqdm

from sampling_methods import point_to_surf_distance_wSign, shapediameter_sampling_cell, volatile_shapediameter_sampling_cell, volatile_shapediameter_sampling_nonormal

def create_data(filename,time_sequence): 
    print(filename)
    do_alignment = True
    sample_points = True
    create_conditions = False

    #list_of_fileids = 'F:/DATA/LALAA/LA_ids_debug.txt'

    #mesh_base = 'F:/DATA/LALAA_faketime/mesh/'
    #align_base =  'F:/DATA/LALAA_faketime/aligned_mesh/'
    #sd_path = 'F:/DATA/LALAA/shape_diameter/'
    #sample_base = 'F:/DATA/LALAA_faketime/samples/'
    mesh_base = 'D:/DTUTeams/Kristine2023/CFA2/4d_sampling/mesh/'
    align_base =  'D:/DTUTeams/Kristine2023/CFA2/4d_sampling/aligned_mesh/'
    sd_path = 'D:/DTUTeams/Kristine2023/CFA2/4d_sampling/shape_diameter/'
    sample_base = 'D:/DTUTeams/Kristine2023/CFA2/4d_sampling/samples/'

    align_to = 'CFA-2_0016_st_2_time'
    align_time = "0"

    # Start for-loop
    #filename = fileids[1]

    sf = []
    #print(filename)
    #norm_file = align_base + filename+'.vtk'

    #time_sequence = np.arange(n_timesteps)
    time_sequence = time_sequence[time_sequence != align_time]
    time_sequence = np.concatenate(([align_time],time_sequence))
    align_time = align_time.zfill(2)

    for i in time_sequence:
        i = str(i).zfill(2)
        time_filename = filename + "_" + i
        norm_file = align_base + time_filename+'.vtk'

        if do_alignment:
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(mesh_base + time_filename + ".vtk")
            reader.Update()
            pd = reader.GetOutput()   
            
            # translate to COM
            if i == align_time: 
                com_filter = vtk.vtkCenterOfMass()
                com_filter.SetInputData(pd)
                com_filter.SetUseScalarsAsWeights(False)
                com_filter.Update()     
                com = com_filter.GetCenter()
                
                t1 = vtk.vtkTransform()
                t1.Translate([-c for c in com])
            
            trans_filter = vtk.vtkTransformFilter()
            trans_filter.SetInputData(pd)
            trans_filter.SetTransform(t1)
            trans_filter.Update()
            pd_transformed = trans_filter.GetOutput()
                
            # Save source pd for future iterations
            if time_filename == align_to + "_" + str(i): 
                writer = vtk.vtkPolyDataWriter()
                writer.SetFileName(os.path.join(os.path.split(mesh_base)[0],"source_pd.vtk"))
                writer.SetInputData(pd_transformed)
                writer.Write()
                
                pd_transformed1 = pd_transformed
                
            else:
                # Load the source pd
                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(os.path.join(os.path.split(mesh_base)[0],"source_pd.vtk"))
                reader.Update()
                source_pd = reader.GetOutput()

                # Find ICP alignment in time=align_time
                if i == align_time: 
                    icp = vtk.vtkIterativeClosestPointTransform()
                    icp.SetSource(pd_transformed)
                    icp.SetTarget(source_pd)    
                    icp.GetLandmarkTransform().SetModeToRigidBody()                   
                    icp.Modified()
                    icp.Update()

                # Apply ICP transform at all timesteps
                TransformFilter = vtk.vtkTransformPolyDataFilter()
                TransformFilter.SetInputData(pd_transformed)
                TransformFilter.SetTransform(icp)
                TransformFilter.Update()
                pd_transformed1 = TransformFilter.GetOutput()
                        
            
            #Save intermediate (for debugging only)
            # debug_filename = 'F:/DATA/LALAAPV/aligned/'+filename+'.vtk'   
            # writer = vtk.vtkPolyDataWriter()
            # writer.SetFileName(debug_filename)
            # writer.SetInputData(pd_transformed1)
            # writer.Write()
            
            #%% Normalize surface to unit-sphere       
            np_points = vtk_to_numpy(pd_transformed1.GetPoints().GetData())
            scale_factor = [1/np.max(np.sqrt(np.sum(np_points**2,1)))]*3
            sf.append(scale_factor)
            #np.min(np.array(sf)[:,0])
            scale_factor = (0.01178938279155948*0.8,)*3 #Within [-0.5:0.5] in hyperdiffusion and with a margin (80%)
            
            t2 = vtk.vtkTransform()
            t2.Scale(scale_factor)
            
            scale_filter = vtk.vtkTransformFilter()
            scale_filter.SetInputData(pd_transformed1)
            scale_filter.SetTransform(t2)
            scale_filter.Update()
            pd_scale = scale_filter.GetOutput()
            
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(norm_file)
            writer.SetInputData(pd_scale)
            writer.Write()


    #%% Sample points with shape diameter function (NUDF)
    if sample_points:
        os.makedirs(sample_base + "/" + filename,exist_ok=True)
        for j,i in enumerate(time_sequence):
            i = str(i).zfill(2)
            time_filename = filename + "_" + str(i)
            norm_file = align_base + time_filename+'.vtk'

            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(norm_file)
            reader.Update()
            surf = reader.GetOutput()   
            
            ### sample_uniform_points
            uni_size = 10000
            uni_space = 1
            uniform_points = np.vstack((np.random.uniform(-uni_space,uni_space,uni_size),np.random.uniform(-uni_space,uni_space,uni_size),np.random.uniform(-uni_space,uni_space,uni_size))).T
        
            uniform_dist = point_to_surf_distance_wSign(uniform_points,surf)
            
            ### Sample shape diameter points
            scale_factor = (0.01178938279155948*0.5*0.8,)*3
            sigma = (5*scale_factor[0]/2)
            in_points, out_points, in_dist, out_dist = volatile_shapediameter_sampling_nonormal([norm_file,2*sigma,50000,100,0.15,sd_path + time_filename])
            # file, sample_sigma, n_points, volatile_factor, volatile_threshold, sd_path
            
            point_array = np.concatenate([uniform_points,in_points,out_points],axis=0)
            sdf_array = np.concatenate([uniform_dist,in_dist,out_dist],axis=0)
                    
            # Save points            
            #print("Inside: ", np.sum(sdf_array<0))
            #print("Outside: ", np.sum(sdf_array>0))
            time_array = np.expand_dims(np.ones(point_array.shape[0])*int(i),1)
            full_array = np.concatenate([point_array,np.expand_dims(sdf_array,1),time_array],axis=1)

            # Save samples
            np.savez(sample_base + "/" + filename + "/samples_"+i+".npz", pos=full_array[full_array[:,3]>0,:], neg=full_array[full_array[:,3]<0,:])

    if create_conditions: 
        condition_vector = np.zeros(5)
        np.savez(sample_base + "/" + filename + "/conditions.npz", cond=condition_vector)

        
    
if __name__ == "__main__":
    list_of_fileids = "D:/DTUTeams/Kristine2023/CFA2/patient_ids.txt"
    #list_of_fileids = "D:/DTUTeams/Kristine2023/CFA3/patient_ids.txt"
    #list_of_fileids = 'F:/DATA/LALAA_faketime/LA_ids_full.txt'
    #list_of_fileids = 'F:/DATA/LALAA/LA_ids_faketime.txt'

    
    align_to = 'CFA-2_0016_st_2_time'
    fileids = []
    f =  open(list_of_fileids,"r")
    for x in f:
        if not (x.strip() == align_to):
            fileids.append(x.strip())

    #fileids.insert(0,align_to)

    time_sequence = np.arange(0,100,5) 
    create_data(align_to, time_sequence = time_sequence )

    # Multiprocessing
    args = list(product(fileids, [time_sequence])) 
    with Pool(16) as p:
      r = list(tqdm.tqdm(p.starmap(create_data, args), total=len(fileids)))

    
