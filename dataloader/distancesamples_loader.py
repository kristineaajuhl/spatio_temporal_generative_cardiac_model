import torch
import torch.utils.data
import numpy as np
#from tqdm import tqdm
import os

class DistanceSamplesLoader_wConditions(torch.utils.data.Dataset):
    def __init__(self, data_path, split_file=None,test=False):
        super().__init__()

        self.test = test
        if self.test == False:
            #self.time_sequence = np.arange(5)
            self.time_sequence = np.arange(0,100,5)
        else: 
            self.time_sequence = np.array([0])  # Timeframe to fit to for test-time optimization
            # TODO: Make the numbers here definable in the config-file or automatically changing

        self.base_paths = self.get_base_paths(data_path,split_file)
        
        print("dataset len: ", len(self.base_paths))


    def __len__(self):
        return len(self.base_paths)

    def __getitem__(self,index):
        if self.test: 
            n_samples = 8000 # Samples for the given time frame
        else: 
            n_samples = 800 # Samples per time frame

        # Load coordinate-distance pairs from .npz files             
        coord = np.zeros((n_samples*self.time_sequence.shape[0],4))
        dist = np.zeros((n_samples*self.time_sequence.shape[0]))
        for i,time_step in enumerate(self.time_sequence): 
            sample_path = self.base_paths[index] + "/samples_" + str(time_step).zfill(2) + ".npz"
            sample1 = np.load(sample_path)
            sample = np.concatenate((sample1["pos"],sample1["neg"]))
            selected_indices = np.random.choice(sample.shape[0], n_samples, replace=False)
            sampled_time = sample[selected_indices,4]/100
            coord[i*n_samples:(i+1)*n_samples,:] = np.concatenate([sample[selected_indices,0:3],np.expand_dims(sampled_time,1)],axis=1)
            dist[i*n_samples:(i+1)*n_samples] = sample[selected_indices,3]

        # Load demographic data
        condition_path = self.base_paths[index] + "/conditions.npz"
        condition = np.load(condition_path)["cond"]
        
        return {
            "coord" : coord,
            "dist" : dist,
            "index" : index,
            "condition" : condition
        }

    # # LALAA - one file
    # def XX__getitem__(self, index):
    #     if not self.test: ## TRAIN!!
    #         # During training (default)
    #         #n_times = 5
    #         #time_step = np.random.randint(n_times)
    #         #condition = np.eye(n_times)[int(time_step)]
    #         #load_path = os.path.join(self.distance_sample_paths[index],self.instance_names[index] + "_"+str(time_step) + ".npz")
            
    #         sample1 = np.load(self.distance_sample_paths[index])
    #         sample = np.concatenate((sample1["pos"],sample1["neg"]))
    #         selected_indices = np.random.choice(sample.shape[0], 16384, replace=False)
    #         coord = np.concatenate([sample[selected_indices,0:3],np.expand_dims(sample[selected_indices,4],1)],axis=1)
    #         dist = sample[selected_indices,3]

    #         condition = np.load(self.conditions_paths[index])["cond"]
            
    #     else: ### TEST !!!
    #         # During testing / test-time-optimization
    #         print("OBS! Test loader not updated for time-varying things")

    #         # time_step = self.instance_names[index][-1::]
    #         # condition =np.eye(5)[int(time_step)]

    #         # load_path = os.path.join(self.distance_sample_paths[index],self.instance_names[index] + ".npz")
    #         # sample1 = np.load(load_path)
    #         # sample = np.concatenate((sample1["pos"],sample1["neg"]))
    #         # selected_indices = np.random.choice(sample.shape[0], 16384, replace=False)
    #         # coord = sample[selected_indices,0:3]
    #         # dist = sample[selected_indices,3]

    #         sample1 = np.load(self.distance_sample_paths[index])
    #         sample = np.concatenate((sample1["pos"],sample1["neg"]))
    #         selected_indices = np.random.choice(sample.shape[0], 16384, replace=False)
    #         coord = np.concatenate([sample[selected_indices,0:3],np.expand_dims(sample[selected_indices,4],1)],axis=1)
    #         dist = sample[selected_indices,3]

    #         condition = np.load(self.conditions_paths[index])["cond"]

    #     return {
    #         "coord" : coord,
    #         "dist" : dist,
    #         "index" : index,
    #         "condition" : condition
    #     }

    

    # def get_distance_sample_paths_wnames(self,data_source, split, extention=""):
    #     file_path = []
    #     filenames = []
    #     for dataset in split: # dataset = "acronym" 
    #         for class_name in split[dataset]:
    #             for instance_name in split[dataset][class_name]:
    #                 instance_filename = os.path.join(data_source, instance_name + extention)
    #                 if not os.path.exists(instance_filename):
    #                     print("Not a path: ", instance_filename)
    #                     continue
    #                 #files.append( torch.from_numpy(np.loadtxt(instance_filename)).float() )
    #                 file_path.append(instance_filename)
    #                 filenames.append(instance_name)
    #     return file_path, filenames
    

    # def get_distance_sample_paths_testtime(self,data_source, split, extention=""):
    #     file_path = []
    #     filenames = []
    #     for dataset in split: # dataset = "acronym" 
    #         for class_name in split[dataset]:
    #             for instance_name in split[dataset][class_name]:
    #                 instance_filename = os.path.join(data_source, instance_name[:-2] + extention)
    #                 if not os.path.exists(instance_filename):
    #                     print("Not a path: ", instance_filename)
    #                     continue
    #                 #files.append( torch.from_numpy(np.loadtxt(instance_filename)).float() )
    #                 file_path.append(instance_filename)
    #                 filenames.append(instance_name)
    #     return file_path, filenames
    
    def get_base_paths(self,data_source, split, extention=""):
        file_path = []
        for dataset in split: # dataset = "acronym" 
            for class_name in split[dataset]:
                for instance_name in split[dataset][class_name]:
                    instance_filename = os.path.join(data_source, instance_name, extention)
                    if not os.path.exists(instance_filename):
                        print("Not a path: ", instance_filename)
                        continue
                    #files.append( torch.from_numpy(np.loadtxt(instance_filename)).float() )
                    file_path.append(instance_filename)
        return file_path

    # def get_paths(self,data_source, split, extention="samples.npz"):
    #     file_path = []
    #     for dataset in split: # dataset = "acronym" 
    #         for class_name in split[dataset]:
    #             for instance_name in split[dataset][class_name]:
    #                 instance_filename = os.path.join(data_source, instance_name, extention)
    #                 if not os.path.isfile(instance_filename):
    #                     print("Not a path: ", instance_filename)
    #                     continue
    #                 #files.append( torch.from_numpy(np.loadtxt(instance_filename)).float() )
    #                 file_path.append(instance_filename)
    #     return file_path

    # ESOF
    # def __getitem__(self, index):
    #     sample = np.load(self.distance_sample_paths[index])
    #     selected_indices = np.random.choice(sample["pos"].shape[0], int(15000/2), replace=False)
    #     #selected_indices = np.arange(0,sample.shape[0],100)
    #     coord = np.concatenate((sample["pos"][selected_indices,0:3],sample["neg"][selected_indices,0:3]),)
    #     dist = np.concatenate((sample["pos"][selected_indices,3],sample["neg"][selected_indices,3]))

    #     #coord_a = torch.cat([torch.zeros(10,1),torch.ones(10,1),torch.ones(10,1)*2])
    #     #coord = np.tile(np.arange(0,100)[:, np.newaxis], reps=(1, 3))
    #     #dist = np.arange(-50,50)/500
    #     #dist = torch.ones(selected_indices.shape) * -0.01
        
        
    #     return {
    #         "coord" : coord,
    #         "dist" : dist,
    #         "index" : index
    #     }
    

    # def get_distance_sample_paths(self,data_source, split, extention=".npz"):
    #     file_path = []
    #     for dataset in split: # dataset = "acronym" 
    #         for class_name in split[dataset]:
    #             for instance_name in split[dataset][class_name]:
    #                 instance_filename = os.path.join(data_source, instance_name + extention)
    #                 if not os.path.isfile(instance_filename):
    #                     print("Not a path: ", instance_filename)
    #                     continue
    #                 #files.append( torch.from_numpy(np.loadtxt(instance_filename)).float() )
    #                 file_path.append(instance_filename)
    #     return file_path

