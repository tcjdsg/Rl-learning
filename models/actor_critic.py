import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic
import torch.nn.functional as F
from models.cnn import cnnNet
import torch


class ActorCritic(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,


                 input_dim,
                 hidden_dims,
                 kernel,
                 hidden_dim,

                 num_mlp_layers_actor,
                 hidden_dim_actor,

                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 out_priority_dim,

                 device
                 ):
        super(ActorCritic, self).__init__()
        # job size for problems, no business with network
        self.n_j = n_j
        # machine size for problems, no business with network
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device

        self.feature_extract = cnnNet( input_dim, hidden_dims, hidden_dim, kernel).to(device)
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim*2, hidden_dim_actor, out_priority_dim).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

    def forward(self,x):

        h_priority = self.feature_extract(x=x)
        # prepare policy feature: concat omega feature with global feature

        pi = self.actor(h_priority)

        v = self.critic(h_priority)
        return pi, v


if __name__ == '__main__':
    print('Go home')