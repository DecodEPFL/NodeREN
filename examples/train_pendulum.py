import os
import torch
from torch.utils import data
from data.datasets import Dataset, Pendulum_simulation_given_time_vector
from data.datasets import Pendulum_simulation_given_initial_conditions

def generate_uniform_test_pendulum(x0_vector,model_parameters=[1.5,0.5], t_i=0.0, t_end=10.0, n_steps=100,seed = 0,device ="cpu"):
    testing_dataset, domain = Pendulum_simulation_given_initial_conditions(x0_vector=x0_vector,model_parameters=model_parameters,t_i=t_i,t_end=t_end,n_steps=n_steps,)
    testing_dataset = testing_dataset.to(device)
    return testing_dataset, domain

def generate_pendulum_dataset(n_exp=100,model_parameters=[1.5,0.5], time_vector=torch.zeros(0,0), n_steps=100, save_file = False, destination_folder="./", name_file="pendulum",seed = 0,extension="csv",delimiter=",", batch_size=125,device ="cpu"):

    my_x0s, my_simulations, domain = Pendulum_simulation_given_time_vector(n_exp,model_parameters,time_vector.cpu().detach().numpy(), n_steps, destination_folder, name_file,seed,extension,delimiter,save_file = save_file) 
    #Final shapes:
        # # my_x0s.shape = [n_exp,nx]
        # # my_simulations.shape = [nt, n_exp, ny == 2]
    my_x0s = my_x0s.to(device)
    my_simulations = my_simulations.to(device)
    
    partition = {'train': range(0, n_exp, 2),
            'test': range(1, n_exp, 2)}
    
    training_set = Dataset(partition['train'], my_x0s, my_simulations)
    training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    
    return my_x0s, my_simulations, domain, training_generator, partition
