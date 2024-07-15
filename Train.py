
import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import os
import json

# add paths in model/__init__.py for new models
from models.combined_model import CombinedModel
from dataloader.distancesamples_loader import DistanceSamplesLoader_wConditions

def train():

    # Initialize dataset and dataloader
    split = json.load(open(specs["TrainSplit"], "r"))

    train_dataset = DistanceSamplesLoader_wConditions(specs["data_path"], split)
   
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, num_workers=args.workers,
        drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True
        )

    # Pytorch lightning callbacks
    callback = ModelCheckpoint(dirpath=args.exp_dir, filename='{epoch}', save_top_k=-1, save_last=True, every_n_epochs=specs["log_freq"])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [callback, lr_monitor]

    # wandb setup
    wandb_logger = WandbLogger(project="SpatioTemporalHeart")

    # Define model
    model = CombinedModel(specs)

    # Resuming training
    if args.resume is not None:
        if args.resume=='last':
            ckpt = "{}.ckpt".format(args.resume)  
        else:
            ckpt = "epoch={}.ckpt".format(args.resume)

        resume = os.path.join(args.exp_dir, ckpt)
    else: 
        resume = None

    # Trainer (pytorch lightning)
    trainer = pl.Trainer(accelerator='gpu', devices=-1, precision=32, max_epochs=specs["num_epochs"], callbacks=callbacks, logger=wandb_logger,log_every_n_steps=1)
    trainer.fit(model=model, train_dataloaders=train_dataloader, ckpt_path=resume)
    



if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", default="/experiments/experiment1/",
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r", default=None,
        help="continue from previous saved logs, integer value or 'last'",
    )

    arg_parser.add_argument("--batch_size", "-b", default=4, type=int)
    arg_parser.add_argument( "--workers", "-w", default=4, type=int)

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"])


    train()