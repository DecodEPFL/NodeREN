import os
import torch
from torch.utils import data

from data.datasets import Dataset2D, lines, swiss_roll, circles, checker_board, moons, double_circles, double_moons,letter


def generate_train_test(dataset='swiss_roll', nf=2, data_size=8000, batch_size=125, device= "cpu"):
    """_summary_

    Args:
        -dataset (str, optional): Dataset to generate. Defaults to 'swiss_roll'.
        -nf (int, optional): no. of (augmented) states of the final dataset. Defaults to 2.
        -data_size (int, optional): no. of samples to generate. Defaults to 8000.
        -batch_size (int, optional): Batch size. Defaults to 125.
        -device (str, optional): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...). Defaults to "cpu".

    Returns:
    TODO
        my_data : (torch.tensor)
        my_label: _description_
        domain: _description_
        training_generator: _description_
        partition: (dictionary)
    """
    if dataset == 'swiss_roll':
        data_gen = swiss_roll
    elif dataset == 'circles':
        data_gen = circles
    elif dataset == 'checker_board':
        data_gen = checker_board
    elif dataset == 'moons':
        data_gen = moons
    elif dataset == 'double_circles':
        data_gen = double_circles
    elif dataset == 'double_moons':
        data_gen = double_moons
    elif dataset == 'lines':
        data_gen = lines
    elif dataset == 'letter':
        data_gen = letter
    else:
        print("%s dataset is not yet implemented" % dataset)
        return

    # define data
    train_data_size = 4000
    test_data_size = data_size - train_data_size
    my_data, my_label, domain = data_gen(data_size, nf=nf, noise_std=0)
    partition = {'train': range(0, data_size, 2),
                 'test': range(1, data_size, 2)}
    my_data = my_data.to(device)
    my_label = my_label.to(device)
    training_set = Dataset2D(partition['train'], my_data, my_label)
    training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    return my_data, my_label, domain, training_generator, partition
