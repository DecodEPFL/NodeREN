import os
import torch
from torch.utils import data
from data.Robot_Dataset import Generator_X0s,Dataset_initial_position


def Generate_initial_positions_robots(nx,data_size=100, batch_size=125,n_agents = 2, std = 0.01, num_workers =1, device ='cpu'):

    # define data
    my_x0s = Generator_X0s(data_size,n_agents, std = std).to(device)
    x0s_REN = torch.zeros(data_size,nx).to(device)
    my_x0s = torch.cat((x0s_REN, my_x0s), dim =1).to(device)
    # my_x0s.shape = [n_exp,nx]
    # my_simulations.shape = [nt, n_exp, ny == 2]
    
    
    partition = {'train': range(0, data_size, 2),
                'test': range(1, data_size, 2)}
    training_set = Dataset_initial_position(partition['train'], my_x0s)
    # training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    return my_x0s, training_generator, partition





