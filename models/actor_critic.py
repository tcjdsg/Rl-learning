import torch
import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic
import torch.nn.functional as F
from models.cnn import cnnNet
import numpy as np


# Trick 8: orthogonal initialization

def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

    return layer
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
        self.activate_func = [nn.ReLU(), nn.Tanh()][0]  # Trick10: use tanh

        self.feature_extract = cnnNet( input_dim, hidden_dims, hidden_dim, kernel).to(device)
        self.actorN = MLPActor(num_mlp_layers_actor, hidden_dim, hidden_dim_actor, out_priority_dim).to(device)
        self.criticN = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)
        # use_orthogonal_init =True
        # if use_orthogonal_init:
        #     print("------use orthogonal init------")
        #     # orthogonal_init(self.feature_extract)
        #     orthogonal_init(self.actor)
        #     orthogonal_init(self.critic)

    def actor(self, s):
        h_priority = self.feature_extract.run(s)
        h = self.activate_func(h_priority)
        pi = self.actorN(h)
        return pi
    def critic(self,x):
        h_priority = self.feature_extract.run(x=x)
        h = self. activate_func(h_priority)
        v = self.criticN(h)
        return  v




if __name__ == '__main__':

    print('Go home')