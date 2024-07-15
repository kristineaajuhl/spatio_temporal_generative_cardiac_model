import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import os
import json
import numpy as np

# add paths in model/__init__.py for new models
from models.combined_model import CombinedModel
from dataloader.distancesamples_loader import DistanceSamplesLoader_wConditions
from models.utils import convert_sdf_samples_to_ply, save_sdf_samples_as_nifti


def reconstruct_timeseries_from_latent_code(model,latent_code,retrieval_res,base_path,filename,t): 
    sdf_filename_out = os.path.join(base_path,"reconstruction"+filename+".nii")
    vtk_filename_out = os.path.join(base_path,"reconstruction"+filename+".ply")

    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (retrieval_res - 1)

    # Predict 
    pred_sdf = model.reconstruct_from_latent_code_time(latent_code,retrieval_res, t=t/100)
    #print(pred_sdf)
    print(torch.min(pred_sdf))
    print(torch.max(pred_sdf))
    pred_sdf = torch.flip(pred_sdf,[0,1])

    if torch.min(pred_sdf) > 0 or torch.max(pred_sdf) < 0: 
        print("There is no zero-level in the distance field - Could not reconstruct")

    else:
        save_sdf_samples_as_nifti(pred_sdf.to('cpu'),voxel_origin,voxel_size,sdf_filename_out)
        convert_sdf_samples_to_ply(pred_sdf.to('cpu'),voxel_origin,voxel_size,vtk_filename_out,scale=2)
   


@torch.no_grad()
def test(): 
    # Load model from checkpoint
    ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
    resume = os.path.join(args.exp_dir, ckpt)
    model = CombinedModel.load_from_checkpoint(resume, specs=specs).cuda().eval()
    model.eval()


    if args.task == "sequence_completion_training":
        base_path = os.path.join(args.exp_dir,"sequence_completion_training")
        os.makedirs(base_path, exist_ok=True)
        number = 0

        # Get filename
        
        train_files = json.load(open(specs["TrainSplit"]))["acronym"]["LALAA"]
        shape_code = model.lat_vecs.weight.data[number]
        id_name = train_files[number]
        condition = torch.tensor(np.load(os.path.join(specs["data_path"],id_name,"conditions.npz"))["cond"]).cuda()
        
        condition_code = model.condition_model(condition)
        concat_code =  torch.cat([condition_code,shape_code],axis=0)

        time_steps = np.arange(0,100,5)
        for i in time_steps: 
            filename = id_name + "_" + str(i)

            reconstruct_timeseries_from_latent_code(model,concat_code,args.retrieval_res,base_path,filename,t=i)

    elif args.task == "sequence_completion_test":
        base_path = os.path.join(args.exp_dir,"sequence_completion_test")
        os.makedirs(base_path, exist_ok=True) 

        test_epochs = 800                    

        # Initialize dataset and dataloader
        test_split = json.load(open(specs["TestSplit"], "r"))

        test_dataset = DistanceSamplesLoader_wConditions(specs["data_path"], test_split,test=True)
        n_test = len(test_dataset)

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1, num_workers=1,
            drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True
            )

        # Pytorch lightning callbacks
        callback = ModelCheckpoint(dirpath=args.exp_dir, filename='{test_time}', save_top_k=-1, save_last=False, every_n_epochs=test_epochs)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [callback, lr_monitor]

        # wandb setup
        wandb_logger = WandbLogger(project="SpatioTemporalHearts")
        
        # Load the trained model
        if args.resume=='last':
            ckpt = "{}.ckpt".format(args.resume)  
        else:
            ckpt = "epoch={}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
        trained_state_dict = torch.load(resume)["state_dict"]

        # Define model
        model = CombinedModel(specs,test=True)
        model.load_state_dict(trained_state_dict)
        
        # Trainer (pytorch lightning)
        trainer = pl.Trainer(accelerator='gpu', devices=-1, precision=32, max_epochs=specs["num_epochs"]+test_epochs, callbacks=callbacks, logger=wandb_logger,log_every_n_steps=1)
        trainer.fit(model=model, train_dataloaders=test_dataloader, ckpt_path=resume)

        # Reconstruct the time sequence
        model.eval()
        model = model.to("cuda:0")
        time_steps = np.arange(0,100,5)
        for sample_idx in np.arange(n_test):
            id_name = test_split["acronym"]["LALAA"][sample_idx]
            shape_code = model.lat_vecs.weight.data[sample_idx,:]

            condition = torch.tensor(np.load(specs["data_path"] + id_name + "/conditions.npz")["cond"]).cuda()
            #condition = torch.zeros(7).cuda()#x["condition"]
            condition_code = model.condition_model(condition)
            concat_code =  torch.cat([condition_code,shape_code],axis=0)

            np.savez(os.path.join(base_path,id_name + "latent.npz"),code = np.array(concat_code.cpu()))

            for i in time_steps: 
                filename = id_name + "_" + str(i)

                reconstruct_timeseries_from_latent_code(model,concat_code,args.retrieval_res,base_path,filename,t=i)

    elif args.task == "sequence_generation":
        base_path = os.path.join(args.exp_dir,"sequence_generation")
        os.makedirs(base_path, exist_ok=True)

        n_examples = 5  # Number of examples for given condition

        # Fit distribution to shape-space and sample
        data = model.lat_vecs.weight.data.detach().cpu()
        estimated_mean = data.mean(dim=0)
        estimated_covariance_matrix = torch.matmul((data - estimated_mean).T, (data - estimated_mean)) / data.size(0)
        multivariate_normal = torch.distributions.MultivariateNormal(estimated_mean, estimated_covariance_matrix)

        # Define conditions
        gender = 0
        age = np.array([0,1,0,0,0])
        bp = (130 - 80) / (200 - 80)    # OBS! Normalization should be the same as in the dataloader
        condition = torch.tensor(np.concatenate(([gender],np.squeeze(age),[bp]))).cuda()

        condition_code = model.condition_model(condition)

        for example in np.arange(n_examples): 
            shape_code = multivariate_normal.sample((1,)).squeeze().cuda()
            concat_code =  torch.cat([condition_code,shape_code],axis=0)

            np.savez(os.path.join(base_path,str(example) + "_latent.npz"),code = np.array(concat_code.cpu()))

            time_steps = np.arange(0,100,5)
            #time_steps = np.array([0])
            for i in time_steps: 
                    filename = str(example) + "_" + str(i).zfill(2)

                    reconstruct_timeseries_from_latent_code(model,concat_code,args.retrieval_res,base_path,filename,t=i)

    elif args.task == "sequence_generation_similarconditions":
        base_path = os.path.join(args.exp_dir,"sequence_generation_similarconditions")
        os.makedirs(base_path, exist_ok=True)

        # DP it for all
        train_filenames = json.load(open(specs["TrainSplit"], "r"))["acronym"]["LALAA"]
        test_filenames = json.load(open(specs["TestSplit"], "r"))["acronym"]["LALAA"]
        all_filenames = test_filenames#train_filenames + test_filenames

        n_examples = 1  # Number of examples per condition

        # Fit distribution to shape-space and sample
        data = model.lat_vecs.weight.data.detach().cpu()
        estimated_mean = data.mean(dim=0)
        estimated_covariance_matrix = torch.matmul((data - estimated_mean).T, (data - estimated_mean)) / data.size(0)
        multivariate_normal = torch.distributions.MultivariateNormal(estimated_mean, estimated_covariance_matrix)

        time_steps = np.arange(0,100,5)
        for id_name in all_filenames: 
            condition = torch.tensor(np.load(specs["data_path"] + id_name + "/conditions.npz")["cond"]).cuda()
            condition_code = model.condition_model(condition)

            for ex in np.arange(n_examples):
                shape_code = multivariate_normal.sample((1,)).squeeze().cuda()
                concat_code =  torch.cat([condition_code,shape_code],axis=0)

                for i in time_steps: 
                    filename = id_name + "_" + str(ex).zfill(2) + "_" + str(i).zfill(2)

                    reconstruct_timeseries_from_latent_code(model,concat_code,args.retrieval_res,base_path,filename,t=i)

        return

    else: 
        print(args.task + "is not a valid task!")
        
    return    

if __name__ == "__main__": 
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", default="/experiments/Test_training/",
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r", default="last",
        help="Which epoch to test",
    )
    arg_parser.add_argument(
        "--task", "-t", default="sequence_completion_test",
        help="Task to test",
    )
    arg_parser.add_argument("--retrieval_res", "-res", default=128, type=int)


    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"])

    test()
