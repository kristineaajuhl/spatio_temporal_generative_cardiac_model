import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionModel(nn.Module):
    def __init__(self, dim_in, depth, hidden_dim, dim_out, norm=True): 
        super().__init__()
        norm_fn = lambda: nn.LayerNorm(hidden_dim) if norm else nn.Identity()

        layers = [nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.SiLU(),
            norm_fn()
        )]

        for _ in range(depth - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                norm_fn()
            ))

        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

        #self.fc_mu = nn.Linear(dim_out, dim_out) 
        #self.fc_var = nn.Linear(dim_out, dim_out) 

    def forward(self, data):
        result = self.net(data.float())
        return result

        #mu = self.fc_mu(result)
        #log_var = self.fc_var(result)

        #return mu, log_var
        
