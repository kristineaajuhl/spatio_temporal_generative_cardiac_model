import torch
import torch.utils.data 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import random
import numpy as np
import math
import socket
import sys


#if socket.gethostname() == 'DESKTOP-UOQS0M5': 
#    sys.path.append("D:/DTUTeam/Kristine2023/NoDiffusion_4D/models/")
#else:     
#    sys.path.append("H:/NoDiffusion_time/models/")
#from models.latent_model import LatentModel
from models.decoder_arch import SdfDecoder
from models.condition_arch import ConditionModel

class CombinedModel(pl.LightningModule):
    def __init__(self, specs, test=False):
        super().__init__()

        self.specs = specs
        self.test = test

        if not self.test: 
            # SDF model related specs
            self.latent_size = specs["latent_specs"]["latent_size"]
            self.shape_latent_size = specs["latent_specs"]["shape_latent_size"]
            self.condition_latent_size = specs["latent_specs"]["condition_latent_size"]
            self.n_conditions = specs["latent_specs"]["n_conditions"]
            self.num_examples = specs["latent_specs"]["num_examples"]
            self.CodeInitStdDev = specs["latent_specs"]["CodeInitStdDev"]
            self.reg_weight = specs["latent_specs"]["CodeRegularizationLambda"]
            self.cond_reg_weight = specs["latent_specs"]["ConditionRegularizationLambda"]

            self.sdf_lr = specs["latent_specs"]["sdf_lr"]
        
            self.dims = specs["latent_model_specs"]["dims"]
            self.dropout = specs["latent_model_specs"]["dropout"]
            self.dropout_prob = specs["latent_model_specs"]["dropout_prob"]
            self.norm_layers = specs["latent_model_specs"]["norm_layers"]
            self.latent_in = specs["latent_model_specs"]["latent_in"]

            self.clamp_value = specs["latent_specs"]["clamp_value"]
            if self.clamp_value: 
                self.minV = -self.clamp_value
                self.maxV = self.clamp_value

            self.loss_l1 = nn.L1Loss(reduction='mean')

            # Initialize models and alike
            self.sdf_model = SdfDecoder(latent_size=self.latent_size,dims=self.dims,dropout=self.dropout,dropout_prob=self.dropout_prob,
                                    norm_layers = self.norm_layers, latent_in = self.latent_in).requires_grad_(True)
            
            self.lat_vecs = torch.nn.Embedding(self.num_examples, self.shape_latent_size, max_norm=None).requires_grad_(True)

            self.condition_model = ConditionModel(dim_in = self.n_conditions, depth = 2, hidden_dim=128,dim_out=self.condition_latent_size).requires_grad_(True)


            torch.nn.init.normal_(
                self.lat_vecs.weight.data,
                0.0,
                self.CodeInitStdDev / math.sqrt(self.shape_latent_size),
            )
            self.sdf_model.train()
            self.lat_vecs.train()
            self.condition_model.train()
        
        if self.test: 
            # SDF model related specs
            self.latent_size = specs["latent_specs"]["latent_size"]
            self.shape_latent_size = specs["latent_specs"]["shape_latent_size"]
            self.condition_latent_size = specs["latent_specs"]["condition_latent_size"]
            self.n_conditions = specs["latent_specs"]["n_conditions"]
            self.num_examples = specs["latent_specs"]["num_examples"]                           # TODO this is super hacky but works as long as num_test < num_train
            self.CodeInitStdDev = specs["latent_specs"]["CodeInitStdDev"]
            self.reg_weight = specs["latent_specs"]["CodeRegularizationLambda"]
            self.cond_reg_weight = specs["latent_specs"]["ConditionRegularizationLambda"]

            self.sdf_lr = 5e-3# specs["latent_specs"]["sdf_lr"]
        
            self.dims = specs["latent_model_specs"]["dims"]
            self.dropout = specs["latent_model_specs"]["dropout"]
            self.dropout_prob = specs["latent_model_specs"]["dropout_prob"]
            self.norm_layers = specs["latent_model_specs"]["norm_layers"]
            self.latent_in = specs["latent_model_specs"]["latent_in"]

            self.clamp_value = specs["latent_specs"]["clamp_value"]
            if self.clamp_value: 
                self.minV = -self.clamp_value
                self.maxV = self.clamp_value

            self.loss_l1 = nn.L1Loss(reduction='mean')

            # Initialize models and alike
            self.sdf_model = SdfDecoder(latent_size=self.latent_size,dims=self.dims,dropout=self.dropout,dropout_prob=self.dropout_prob,
                                    norm_layers = self.norm_layers, latent_in = self.latent_in).requires_grad_(False)
            
            self.lat_vecs = torch.nn.Embedding(self.num_examples, self.shape_latent_size, max_norm=None).requires_grad_(True)

            self.condition_model = ConditionModel(dim_in = self.n_conditions, depth = 2, hidden_dim=128,dim_out=self.condition_latent_size).requires_grad_(False)

            torch.nn.init.normal_(
                self.lat_vecs.weight.data,
                0.0,
                self.CodeInitStdDev / math.sqrt(self.shape_latent_size),
            )
            self.lat_vecs.train()

        
    def training_step(self, x, idx):
        if not self.test: 
            return self.train_latent(x)
        else: 
            return self.optimize_latent_testtime(x)
        


    def configure_optimizers(self):
        if not self.test: 
            params_list = [
                {'params': self.lat_vecs.parameters(), 'lr': self.specs["latent_specs"]['latent_lr']},
                {'params': self.sdf_model.parameters(), 'lr': self.specs["latent_specs"]['sdf_lr']},
                {'params': self.condition_model.parameters(), 'lr': self.specs["latent_specs"]['sdf_lr']}
            ]
            optimizer = torch.optim.Adam(params_list)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.5, last_epoch=-1, verbose=False)
        else: 
            params_list = [
                {'params': self.lat_vecs.parameters(), 'lr': 5e-3},
                {'params': self.sdf_model.parameters(), 'lr': 0},
                {'params': self.condition_model.parameters(), 'lr': 0}
            ]
            optimizer = torch.optim.Adam(params_list)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.9, last_epoch=-1, verbose=False)
        
        return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler 
        }
    

    def train_latent(self,x): 
        # Prepare input
        input = x["coord"].view(-1, x["coord"].size(2))
        distance = x["dist"].view(-1)
        index = x["index"]
        #conditions = torch.zeros((index.shape[0],7)).cuda()
        conditions = x["condition"]

        n_samples = input.shape[0]/len(index)
        index_repeat = index.repeat_interleave(int(n_samples))
        shape_latent = self.lat_vecs(index_repeat)
        #latent_dropout = nn.Dropout(p=0.2)
        #shape_latent = latent_dropout(shape_latent)
        #concat_input = torch.cat([shape_latent,input],axis=1)
        pred_cond = self.condition_model(conditions)        
        condition_latent = pred_cond.repeat_interleave(int(n_samples),axis=0)
        concat_input = torch.cat([condition_latent,shape_latent,input],axis=1)

        # predict
        pred_sdf = self.sdf_model(concat_input).squeeze()

        #print(self.lat_vecs.weight.grad)
        #print(self.sdf_model.lin0.weight.grad)

        # compute loss
        pred_sdf = torch.clamp(pred_sdf,-self.clamp_value,self.clamp_value)
        distance = torch.clamp(distance,-self.clamp_value,self.clamp_value)

        l1_loss = self.loss_l1(pred_sdf,distance)
        reg_loss = torch.sum(torch.norm(self.lat_vecs(index), dim=1))
        logprob_loss = self.cdf_loss(self.lat_vecs(index))
        condition_reg_loss = torch.sum(torch.norm(pred_cond, dim=1))

        #total_loss = self.reg_weight*reg_loss*min(1, self.current_epoch / 100) + l1_loss 
        #total_loss = l1_loss + (self.reg_weight * logprob_loss) * min(1, self.current_epoch / 100) 
        #total_loss = l1_loss + (self.reg_weight * logprob_loss + self.cond_reg_weight * condition_reg_loss) * min(1, self.current_epoch / 100) 
        #total_loss = l1_loss + (self.reg_weight * reg_loss + self.cond_reg_weight * condition_reg_loss) * min(1, self.current_epoch / 100) 
        total_loss = l1_loss + (self.reg_weight * reg_loss) * min(1, self.current_epoch / 100) 

        loss_dict = {
            "total:": total_loss,
            "regularization: ": reg_loss,
            "log_prob: ": logprob_loss,
            "condition_regularization: ": condition_reg_loss,
            "reconstruction: ": l1_loss,
            "pred_min: ": pred_sdf.min(),
            "pred_max: ": pred_sdf.max()
        }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return total_loss
    
    def optimize_latent_testtime(self,x):
        self.sdf_model.eval()
        self.condition_model.eval()

        # Prepare input
        input = x["coord"].view(-1, x["coord"].size(2))
        distance = x["dist"].view(-1)
        index = x["index"]
        #conditions = torch.zeros((index.shape[0],7)).cuda()#x["condition"]
        conditions = x["condition"]
        
        n_samples = input.shape[0]/len(index)
        index_repeat = index.repeat_interleave(int(n_samples))
        shape_latent = self.lat_vecs(index_repeat)
        #concat_input = torch.cat([shape_latent,input],axis=1)
        pred_cond = self.condition_model(conditions)        
        condition_latent = pred_cond.repeat_interleave(int(n_samples),axis=0)
        concat_input = torch.cat([condition_latent,shape_latent,input],axis=1)

        # predict
        pred_sdf = self.sdf_model(concat_input).squeeze()

        #print(self.lat_vecs.weight.grad)
        #print(self.sdf_model.lin0.weight.grad)

        # compute loss
        pred_sdf = torch.clamp(pred_sdf,-self.clamp_value,self.clamp_value)
        distance = torch.clamp(distance,-self.clamp_value,self.clamp_value)

        l1_loss = self.loss_l1(pred_sdf,distance)
        reg_loss = torch.sum(torch.norm(self.lat_vecs(index), dim=1))
        logprob_loss = self.cdf_loss(self.lat_vecs(index))
        condition_reg_loss = torch.sum(torch.norm(pred_cond, dim=1))

        #total_loss = self.reg_weight*reg_loss*min(1, self.current_epoch / 100) + l1_loss 
        total_loss = l1_loss + (self.reg_weight * logprob_loss) * min(1, self.current_epoch / 100) 
        #total_loss = l1_loss + (self.reg_weight * logprob_loss + self.cond_reg_weight * condition_reg_loss) * min(1, self.current_epoch / 100) 

        loss_dict = {
            "total:": total_loss,
            "regularization: ": reg_loss,
            "log_prob: ": logprob_loss,
            "condition_regularization: ": condition_reg_loss,
            "reconstruction: ": l1_loss,
            "pred_min: ": pred_sdf.min(),
            "pred_max: ": pred_sdf.max()
        }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return total_loss

    def reconstruct_from_latent_code_time(self,latent_code,retrieval_res,t):
        self.sdf_model.eval()

        # Create grid
        N = retrieval_res
        x = np.linspace(-0.5, 0.5, N)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        X = X.reshape((np.prod(X.shape),))
        Y = Y.reshape((np.prod(Y.shape),))
        Z = Z.reshape((np.prod(Z.shape),))
        T = np.ones(X.shape)*t

        samples = torch.tensor(np.column_stack((X, Y, Z,T)))

        num_samples = N ** 3
        latent_repeat = latent_code.unsqueeze(dim=0).repeat_interleave(int(num_samples),dim=0)

        batches = int(np.ceil(num_samples/(64**3)))*2
        samples_per_batch = int(num_samples/batches)
        pred_sdf = torch.zeros(num_samples)
        

        print("Number of batches: ",batches)
        for i in range(batches):
            # Concatenate with latent_vector
          
            concat_input = torch.cat([latent_repeat[i*samples_per_batch:(i+1)*samples_per_batch,:],samples[i*samples_per_batch:(i+1)*samples_per_batch,:].to('cuda:0')],axis=1)

            #batch_input = concat_input[i*samples_per_batch:(i+1)*samples_per_batch,:]
            #print(batch_input.shape)
            batch_sdf = self.sdf_model(concat_input).squeeze()
            batch_sdf_cpu = batch_sdf.detach().cpu()
            pred_sdf[i*samples_per_batch:(i+1)*samples_per_batch] = batch_sdf_cpu
        

        # Transform to surface
        pred_sdf = pred_sdf.reshape(N, N, N)
        pred_sdf = pred_sdf.flip(0,1)

        return pred_sdf
    
    def cdf_loss(self,sample):
        mean = torch.tensor(np.zeros(sample.shape[1])).cuda()
        covariance_matrix = torch.eye(sample.shape[1]).cuda()#torch.tensor([[1.0, 0.5], [0.5, 2.0]])
        
        # Multivariate normal distribution
        mvn = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=covariance_matrix)

        # Calculate the log_prob
        cdf_value = torch.abs(mvn.log_prob(sample))

        return torch.mean(cdf_value)
