import os
import torch
from torch.utils import data
from data.datasets import cartpole_simulation, Dataset


def generate_train_test_cartpole(n_exp=100,model_parameters=[1.5,0.5], t_i=0.0, t_end=10.0, n_steps=100, destination_folder="./", name_file="cartpole",seed = 0,extension="csv",delimiter=",", data_size=8000, batch_size=125,device = "cpu"):

    partition = {'train': range(0, data_size, 2),
                'test': range(1, data_size, 2)}
    # define data
    my_x0s, my_simulations, domain = cartpole_simulation(n_exp,model_parameters,t_i, t_end, n_steps, destination_folder, name_file,seed,extension,delimiter) 
    # my_x0s.shape = [n_exp,nx]
    # my_simulations.shape = [nt, n_exp, ny == 2]
    my_x0s = my_x0s.to(device)
    my_simulations = my_simulations.to(device)
    gain = 1.
    my_x0s = gain*my_x0s
    my_simulations= gain* my_simulations
    domain = [element * gain for element in domain]
    partition = {'train': range(0, data_size, 2),
                'test': range(1, data_size, 2)}
    training_set = Dataset(partition['train'], my_x0s, my_simulations)
    training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    return my_x0s, my_simulations, domain, training_generator, partition