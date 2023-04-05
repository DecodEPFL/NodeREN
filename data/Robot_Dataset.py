import torch
import math
from math import pi as pi
from torch.utils import data
import numpy as np
import os
from scipy.integrate import odeint as odeint_sym



class Dataset_initial_position(data.Dataset):

    def __len__(self):
        return len(self.list_IDs)

    def __init__(self, list_IDs, data):
        self.list_IDs = list_IDs
        self.data = data

    def __getitem__(self, index):

        ID = self.list_IDs[index]

        X = self.data[ID, :]

        return X

def Generator_X0s(data_size,n_agents, std):
    # np.random.seed(seed=seed)
    #std = 0.001
    if (n_agents == 2):
        x0 = torch.tensor([-2, -2, 0, 0, 2, -2, 0, 0])
        x0s = x0.repeat(data_size,1)    #now it is a [ data_size, n_agents * 4]
    else:
        #you need to create your ideal initial state ... idk them yet
        raise NameError("Not implemented yet the swarm...")
    
    #let's add some noise to the states
    noise = torch.cat((torch.randn(data_size,2)*std, torch.zeros(data_size,2)), dim=1)
    for i in range(1,n_agents):
        temp = torch.cat((torch.randn(data_size,2)*std, torch.zeros(data_size,2)), dim=1)
        noise = torch.cat((noise,temp), dim=1)
    final_x0s = x0s + noise
    return final_x0s 