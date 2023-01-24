import torch
import math
from math import pi as pi
from torch.utils import data
import numpy as np
import os
from scipy.integrate import odeint as odeint_sym
def _data_shuffle(data2d, label):
    data_size = data2d.shape[0]
    randindex = torch.randperm(data_size)
    data2d = data2d[randindex, :, :]
    label = label[randindex, :]
    return data2d, label


def _data_extension(data2d, nf, input_ch=None, device= "cpu"):
    if nf < 2:
        print("Dimension not valid")
        return
    elif nf % 2 == 1:
        print("Using odd dimension nf")
    data_size = data2d.shape[0]
    if input_ch is not None:
        # input_ch is a list of two elements. The elements indicate where the data enters.
        idx_x = input_ch[0]
        idx_y = input_ch[1]
    else:
        idx_x = 0
        idx_y = nf - 1
    data2d = torch.cat((torch.zeros(data_size, idx_x - 0, 1,device = device),
                        data2d[:, 0:1, :],
                        torch.zeros(data_size, idx_y - idx_x - 1, 1,device = device),
                        data2d[:, 1:2, :],
                        torch.zeros(data_size, nf - 1 - idx_y, 1,device = device)), 1)
    return data2d


def swiss_roll(dataSize, shuffle=True, nf=2, noise_std=0, input_ch=None):

    data = torch.zeros(dataSize, 2, 1)
    label = torch.ones(dataSize, 1)
    label[math.floor(dataSize/2):, :] = 0

    r1 = torch.linspace(0, 1, math.ceil(dataSize/2))
    r2 = torch.linspace(0.2, 1.2, math.ceil(dataSize/2))
    theta = torch.linspace(0, 4*math.pi-4*math.pi/math.ceil(dataSize/2), math.ceil(dataSize/2))
    data[0:math.ceil(dataSize/2), 0, 0] = r1 * torch.cos(theta)
    data[0:math.ceil(dataSize/2), 1, 0] = r1 * torch.sin(theta)
    data[math.floor(dataSize / 2):, 0, 0] = r2 * torch.cos(theta)
    data[math.floor(dataSize / 2):, 1, 0] = r2 * torch.sin(theta)

    if shuffle:
        data, label = _data_shuffle(data, label)

    if nf != 2:
        data = _data_extension(data, nf, input_ch)

    if noise_std:
        for i in range(2):
            data[:, i, 0] = data[:, i, 0] + noise_std*torch.randn(dataSize)
    
    domain = [-1.2, 1.2, -1.2, 1.2]
    return data, label, domain

def lines(dataSize, shuffle=True, nf=2, noise_std=0, input_ch=None):

    data = torch.zeros(dataSize, 2, 1)
    label = torch.ones(dataSize, 1)
    label[math.floor(dataSize/2):, :] = 0
    b1 = 1.1 #torch.randn(1,1)
    b2 = -0.7
    m1 = 0.33
    m2 = -0.3
    x = torch.linspace(-1.95,1.95,math.ceil(dataSize/2))
    y1 = m1*x+b1*torch.ones(math.ceil(dataSize/2))
    y2 = m2*x+b2*torch.ones(math.ceil(dataSize/2))
    data[0:math.ceil(dataSize/2), 0, 0] = x
    data[0:math.ceil(dataSize/2), 1, 0] = y1
    data[math.floor(dataSize / 2):, 0, 0] = x
    data[math.floor(dataSize / 2):, 1, 0] = y2

    if shuffle:
        data, label = _data_shuffle(data, label)

    if nf != 2:
        data = _data_extension(data, nf, input_ch)

    if noise_std:
        for i in range(2):
            data[:, i, 0] = data[:, i, 0] + noise_std*torch.randn(dataSize)
    
    domain = [-2., 2., -2., 2.]
    return data, label, domain


def letter(dataSize, shuffle=True, nf=2, noise_std=0, input_ch=None):

    data = torch.zeros(dataSize, 2, 1)
    label = torch.zeros(dataSize, 1)
    # label[math.floor(dataSize/2):, :] = 0
    b1 = 0.12 #torch.randn(1,1)
    m1 = 1.5
    # dim = math.floor(math.sqrt(dataSize))
    # x1 = torch.linspace(-1.0,1.0,dim)
    # x2 = torch.linspace(-1.0,1.0,dim)
    # for i in range(dim):
    #     for j in range(dim):
    #         data[i+j,0,0] = x1[i]
    #         data[i+j,1,0] = x2[j]
    #         y = x2[j] - torch.abs(m1*x1[j])-b1
    #         label[i+j,:] = torch.ge(y,0.)
    # resto = dataSize - dim*dim
    # for i in range(resto):
    #     x1= torch.randn(1)*0.5
    #     x2= torch.randn(1)*0.5
    #     y = x2 - torch.abs(m1*x1)-b1
    #     data[-1-i,0,0] = x1
    #     data[-1-i,1,0] = x2
    #     label[-1-i,:] = torch.ge(y,0.)
    for i in range(dataSize):
        x1= torch.randn(1)*0.5 + 0.0
        x2= torch.randn(1)*0.5 + 0.9
        y = x2 - torch.abs(m1*x1)-b1
        data[i,0,0] = x1
        data[i,1,0] = x2
        label[i,:] = torch.ge(y,0.)

    if shuffle:
        data, label = _data_shuffle(data, label)

    if nf != 2:
        data = _data_extension(data, nf, input_ch)

    if noise_std:
        for i in range(2):
            data[:, i, 0] = data[:, i, 0] + noise_std*torch.randn(dataSize)
    
    domain = [-2., 2., -2., 2.]
    return data, label, domain

def moons(dataSize, shuffle=True, nf=2, noise_std=0, input_ch=None):

    data = torch.zeros(dataSize, 2, 1)
    label = torch.zeros(dataSize, 1)

    for i in range(int(dataSize/2)):
        theta = torch.tensor(i/int(dataSize/2)*3.14)

        label[i, :] = 0
        data[i, :, :] = torch.tensor(
            [[torch.cos(theta) + 0.3*(torch.rand(1)-0.5)], [torch.sin(theta) + 0.3*(torch.rand(1)-0.5)]])

        label[i+int(dataSize/2), :] = 1
        data[i+int(dataSize/2), :, :] = torch.tensor(
            [[torch.ones(1) - torch.cos(theta) + 0.3*(torch.rand(1)-0.5)], [torch.ones(1)*0.5 - torch.sin(theta) + 0.3*(torch.rand(1)-0.5)]])

    if shuffle:
        data, label = _data_shuffle(data, label)

    if nf != 2:
        data = _data_extension(data, nf, input_ch)

    if noise_std:
        for i in range(2):
            data[:, i, 0] = data[:, i, 0] + noise_std*torch.randn(dataSize)

    domain = [-2, 3, -1, 2]
    return data, label, domain


def circles(dataSize, shuffle=True, nf=2, noise_std=0, input_ch=None):

    data = torch.zeros(dataSize, 2, 1)
    label = torch.zeros(dataSize, 1)

    for i in range(int(dataSize/2)):
        theta = torch.tensor(i/int(dataSize/2)*4*3.14)

        r = 1
        label[i, :] = 0
        data[i, :, :] = torch.tensor(
            [[r*torch.cos(theta) + 0.6*(torch.rand(1)-0.5)], [r*torch.sin(theta) + 0.6*(torch.rand(1)-0.5)]])

        r = 2
        label[i+int(dataSize/2), :] = 1
        data[i+int(dataSize/2), :, :] = torch.tensor(
            [[r*torch.cos(theta) + 0.6*(torch.rand(1)-0.5)], [r*torch.sin(theta) + 0.6*(torch.rand(1)-0.5)]])

    if shuffle:
        data, label = _data_shuffle(data, label)

    if nf != 2:
        data = _data_extension(data, nf, input_ch)

    if noise_std:
        for i in range(2):
            data[:, i, 0] = data[:, i, 0] + noise_std*torch.randn(dataSize)

    domain = [-3, 3, -3, 3]
    return data, label, domain


def checker_board(dataSize, shuffle=True, n_clusters=3, nf=2, noise_std=0, input_ch=None):
    n = math.ceil(dataSize/(n_clusters**2))
    ax = torch.linspace(-n_clusters/2, n_clusters/2-1, n_clusters)
    h = 0
    v = 0
    xr = torch.zeros(math.ceil(n_clusters**2/2)*n, 2, 1)
    ir = 0
    xb = torch.zeros(math.floor(n_clusters**2/2)*n, 2, 1)
    ib = 0
    for i in ax:
        for j in ax:
            if math.fmod(h + v, 2) == 0:
                xr[ir:ir+n, 0, 0] = i + torch.rand(n)
                xr[ir:ir+n, 1, 0] = j + torch.rand(n)
                ir = ir + n
            else:
                xb[ib:ib + n, 0, 0] = i + torch.rand(n)
                xb[ib:ib + n, 1, 0] = j + torch.rand(n)
                ib = ib + n
            h = h + 1
        h = 0
        v = v + 1

    data = torch.cat((xr, xb), 0)
    label = torch.cat((torch.ones(xr.size(0), 1), torch.zeros(xb.size(0), 1)), 0)
    data, label = data[0:dataSize, :, :], label[0:dataSize, :]

    if shuffle:
        data, label = _data_shuffle(data, label)

    if nf != 2:
        data = _data_extension(data, nf, input_ch)

    if noise_std:
        for i in range(2):
            data[:, i, 0] = data[:, i, 0] + noise_std*torch.randn(dataSize)

    aux = n_clusters/2 + 0.5
    domain = [-aux, aux, -aux, aux]
    return data, label, domain


def peaks(dataSize, shuffle=True, nf=2, end_value=4, noise_std=0, input_ch=None):

    n = math.ceil(math.sqrt(dataSize))
    x = torch.linspace(-end_value, end_value, n)
    y = torch.zeros(x.size())
    xv, yv = torch.meshgrid([x, x])
    y = 3 * (1 - xv)**2 * torch.exp(-(xv)**2 - (yv+1)**2) - 10 * (xv/5 - xv**3 - yv**5) \
         * torch.exp(-xv**2 - yv**2) - 1/3 * torch.exp(- (xv+1)**2 - yv**2)

    data = torch.cat((xv.contiguous().view([n*n, 1]), yv.contiguous().view([n*n, 1])), 1).unsqueeze(2)
    label = y.contiguous().view([n*n, 1])

    if shuffle:
        data, label = _data_shuffle(data, label)

    if nf != 2:
        data = _data_extension(data, nf, input_ch)

    if noise_std:
        #for i in range(2):
        #    data[:, i, 0] = data[:, i, 0] + noise_std*torch.randn(data.size(0))
        label_noise = label*0
        label_noise[:, 0] = label[:, 0] + noise_std*torch.randn(label.size(0))
        return data[0:dataSize, :, :], label_noise[0:dataSize, :], label[0:dataSize, :]

    domain = [-4, 4, -4, 4]
    return data[0:dataSize, :, :], label[0:dataSize, :], domain


def double_circles(dataSize, shuffle=True, nf=2, noise_std=0, input_ch=None):

    data = torch.zeros(dataSize, 2, 1)
    label = torch.zeros(dataSize, 1)

    for i in range(int(dataSize/4)):
        theta = torch.tensor(i/int(dataSize/4)*4*3.14)

        r = 1
        label[i, :] = 0
        data[i, :, :] = torch.tensor(
            [[r * torch.cos(theta) + 0.6 * (torch.rand(1) - 0.5)],
             [r * torch.sin(theta) + 0.6 * (torch.rand(1) - 0.5)]])

        r = 2
        label[i + int(dataSize/4), :] = 1
        data[i + int(dataSize/4), :, :] = torch.tensor(
            [[r * torch.cos(theta) + 0.6 * (torch.rand(1) - 0.5)],
             [r * torch.sin(theta) + 0.6 * (torch.rand(1) - 0.5)]])

        r = 3
        label[i + int(2*dataSize/4), :] = 0
        data[i ++ int(2*dataSize/4), :, :] = torch.tensor(
            [[r * torch.cos(theta) + 0.6 * (torch.rand(1) - 0.5)],
             [r * torch.sin(theta) + 0.6 * (torch.rand(1) - 0.5)]])

        r = 4
        label[i + int(3*dataSize/4), :] = 1
        data[i + int(3*dataSize/4), :, :] = torch.tensor(
            [[r * torch.cos(theta) + 0.6 * (torch.rand(1) - 0.5)],
             [r * torch.sin(theta) + 0.6 * (torch.rand(1) - 0.5)]])

    if shuffle:
        data, label = _data_shuffle(data, label)

    if nf != 2:
        data = _data_extension(data, nf, input_ch)

    if noise_std:
        for i in range(2):
            data[:, i, 0] = data[:, i, 0] + noise_std*torch.randn(dataSize)
    
    domain = [-5, 5, -5, 5]
    return data, label, domain


def double_moons(dataSize, shuffle=True, nf=2, noise_std=0, input_ch=None):

    data = torch.zeros(dataSize, 2, 1)
    label = torch.zeros(dataSize, 1)

    for i in range(int(dataSize/4)):
        theta = torch.tensor(i/int(dataSize/4)*3.14)

        label[i, :] = 0
        data[i, :, :] = torch.tensor(
            [[torch.cos(theta) + 0.3*(torch.rand(1)-0.5)], [torch.sin(theta) + 0.3*(torch.rand(1)-0.5)]])

        label[i+int(dataSize/4), :] = 1
        data[i+int(dataSize/4), :, :] = torch.tensor(
            [[torch.ones(1) - torch.cos(theta) + 0.3*(torch.rand(1)-0.5)],
             [torch.ones(1)*0.5 - torch.sin(theta) + 0.3*(torch.rand(1)-0.5)]])

        label[i + int(dataSize/2), :] = 0
        data[i + int(dataSize/2), :, :] = torch.tensor(
            [[torch.cos(theta) + 0.3 * (torch.rand(1) - 0.5) + 2*torch.ones(1)],
             [torch.sin(theta) + 0.3 * (torch.rand(1) - 0.5)]])

        label[i + int(dataSize*3/4), :] = 1
        data[i + int(dataSize*3/4), :, :] = torch.tensor(
            [[torch.ones(1) - torch.cos(theta) + 0.3 * (torch.rand(1) - 0.5) + 2*torch.ones(1)],
             [torch.ones(1) * 0.5 - torch.sin(theta) + 0.3 * (torch.rand(1) - 0.5)]])

    if shuffle:
        data, label = _data_shuffle(data, label)

    if nf != 2:
        data = _data_extension(data, nf, input_ch)

    if noise_std:
        for i in range(2):
            data[:, i, 0] = data[:, i, 0] + noise_std*torch.randn(dataSize)
    
    domain = [-2, 5, -1, 2]
    return data, label, domain


class Dataset(data.Dataset):

    def __len__(self):
        return len(self.list_IDs)

    def __init__(self, list_IDs, data, labels):
        self.list_IDs = list_IDs
        self.data = data
        self.labels = labels

    def __getitem__(self, index):

        ID = self.list_IDs[index]

        X = self.data[ID, :]
        y = self.labels[:,ID,:]

        return X, y


class Dataset2D(data.Dataset):

    def __len__(self):
        return len(self.list_IDs)

    def __init__(self, list_IDs, data, labels):
        self.list_IDs = list_IDs
        self.data = data
        self.labels = labels

    def __getitem__(self, index):

        ID = self.list_IDs[index]

        X = self.data[ID, :, :]
        y = self.labels[ID, :]

        return X, y


# MODEL OF THE SYSTEM
def pendulum(z, t, beta, omega):  # define the ode function
    z1, z2 = z
    dz1 = z2
    dz2 = -omega*omega*np.sin(z1)-beta*z2
    dz = np.array([dz1, dz2])
    return dz

def Pendulum_simulation(n_exp=100,model_parameters=[1.5,0.5], t_i=0.0, t_end=10.0, n_steps=100, destination_folder="./", name_file="pendulum",seed = 0,extension="csv",delimiter=","):
    """Simulation of the free evolutions of a non-linear pendulum starting from #n_exp different random initial states with [beta,omega]=model_parameters.
    The resulting (n_steps,2*n_exp) matrix is saved in the "name_file" inside the " destination_folder"

    Args:
        *n_exp (int, optional): number of simulated evolutions. Defaults to 100.\n
        *model_parameters (list, optional): Parameters of the model. model_parameters = [beta,omega]. Please, consider the model designed inside the "Week Update 2". Defaults to [1.5,0.5].\n
        *t_i (float, optional): starting time instant of simulation. Defaults to 0.0.\n
        *t_end (float, optional): ending time instant of simulation. Defaults to 10.0.\n
        *n_steps (int, optional): no. of time steps considered for the simulation. Defaults to 100.\n
        *destination_folder (str): Destination folder of the simulated results. Defaults  to "./"\n
        *name_file (str, optional): Name of the results file. Defaults to "pendulum".\n
        *seed (int, optional): seed used by the library Numpy. Defaults to 100.\n
        *extension (str, optional): Extension of the results file. Defaults to "csv".\n
        *delimiter (str, optional): Delimiter of the results file. Defaults to ",".\n
    Returns:
        matrix: z_real matrix
    """
    np.random.seed(seed=seed)
    if(not(os.path.isdir(destination_folder))):
        os.makedirs(destination_folder)
    real_beta= model_parameters[0]
    real_l = model_parameters[1]  # m  #the length of the pendulum
    g = 9.81  # gravity acceleration
    real_omega_square = g/real_l
    real_omega = np.abs(np.sqrt(g/real_l))
    time_vector = np.linspace(t_i, t_end, n_steps)
    real_args = (real_beta, real_omega)
    csv_file = os.path.join(destination_folder,name_file+"."+extension)
    # if(os.path.exists(csv_file)):
    #     #The file exists so you just need to load it
    #     print('Loading data...')
    #     z_real = np.loadtxt(csv_file, delimiter=delimiter)
    #     print("Data Loaded!")
    #     #Check if they were done using the same number of initial values
    
    # else:
    initial_positions_vector = np.random.uniform(low=-pi/2.0, high=pi/2.0, size=(n_exp,1))
    initial_velocity_vector  = pi/180.0*np.random.randn(n_exp,1)
    z0_vector = np.concatenate((initial_positions_vector,initial_velocity_vector),axis=1)
    z_real = np.zeros((n_steps,n_exp,2))
    max_values = np.amax(np.abs(z0_vector),axis=0)
    #generating the data
    for i in range(n_exp):
        z_temp = odeint_sym(pendulum, z0_vector[i], time_vector, args=real_args)
        z_real[:,i,:] = z_temp
    print("---------------------------------------------------------------------------------------------------------------------------------------")
    print("Finished the simulation phase")
    print("Saving data...")
    z_real_reshaped = z_real.reshape(z_real.shape[0], -1)
    np.savetxt(csv_file, z_real_reshaped, delimiter=delimiter)  # save data
    print("Data Saved!")
    print("---------------------------------------------------------------------------------------------------------------------------------------")
    x0s = torch.tensor(z_real[0,:,:]).float()
    domain = [-max_values[0], + max_values[0],-max_values[1], + max_values[1]]
    return x0s, torch.tensor(z_real).float(),domain


def cartpole(z, t, mp, mc, l, beta):  # define the ode function
    theta, x, theta_dot, x_dot = z
    g = 9.81  # gravity acceleration
    zdot = np.empty((4,))
    zdot[0] = theta_dot
    zdot[1] = x_dot
    # zdot[2] = (g*np.sin(theta)-np.cos(theta)*(mp*l*theta_dot**2*np.sin(theta))/(mc+mp))/l/(4./3.-mp*np.cos(theta)**2/(mc+mp))
    # zdot[3] = (mp*l*(theta_dot**2*np.sin(theta)-zdot[2]*np.cos(theta)))/(mc+mp)
    zdot[3] = (-beta*x_dot+mp*g*np.sin(theta)*np.cos(theta)-mp*l*theta_dot**2*np.sin(theta))/(mc+mp-mp*np.cos(theta)**2)
    zdot[2] = (g*np.sin(theta)+zdot[3]*np.cos(theta))/l
    return zdot

def cartpole_simulation(n_exp=100,model_parameters=[2,10,0.4,1.], t_i=0.0, t_end=10.0, n_steps=100, destination_folder="./", name_file="cartpole",seed = 0,extension="csv",delimiter=","):
    """Simulation of the free evolutions of a non-linear inverted pendulum on a cart.
    The resulting (n_steps,2*n_exp) matrix is saved in the "name_file" inside the " destination_folder"

    Args:
        *n_exp (int, optional): number of simulated evolutions. Defaults to 100.\n
        *model_parameters (list, optional): Parameters of the model. model_parameters = [beta,omega]. Please, consider the model designed inside the "Week Update 2". Defaults to [1.5,0.5].\n
        *t_i (float, optional): starting time instant of simulation. Defaults to 0.0.\n
        *t_end (float, optional): ending time instant of simulation. Defaults to 10.0.\n
        *n_steps (int, optional): no. of time steps considered for the simulation. Defaults to 100.\n
        *destination_folder (str): Destination folder of the simulated results. Defaults  to "./"\n
        *name_file (str, optional): Name of the results file. Defaults to "pendulum".\n
        *seed (int, optional): seed used by the library Numpy. Defaults to 100.\n
        *extension (str, optional): Extension of the results file. Defaults to "csv".\n
        *delimiter (str, optional): Delimiter of the results file. Defaults to ",".\n
    Returns:
        matrix: z_real matrix
    """
    np.random.seed(seed=seed)
    if(not(os.path.isdir(destination_folder))):
        os.makedirs(destination_folder)
    real_mp= model_parameters[0]
    real_mc= model_parameters[1]
    real_l = model_parameters[2]  # m  #the length of the pendulum
    real_beta = model_parameters[3]
    time_vector = np.linspace(t_i, t_end, n_steps)
    real_args = (real_mp, real_mc, real_l,real_beta)
    csv_file = os.path.join(destination_folder,name_file+"."+extension)
    
    # else:
    initial_x_vector = np.random.uniform(low=-real_l, high=real_l, size=(n_exp,1))
    initial_x_dot_vector = np.zeros((n_exp,1))
    initial_theta_vector = np.random.uniform(low=-pi/3.0, high=pi/3.0, size=(n_exp,1))
    # initial_theta_dot_vector  = pi/180.0*np.random.randn(n_exp,1)
    initial_theta_dot_vector  = np.zeros((n_exp,1))
    z0_vector = np.concatenate((initial_theta_vector,initial_x_vector,initial_theta_dot_vector,initial_x_dot_vector,),axis=1)
    z_real = np.zeros((n_steps,n_exp,4))
    max_values = np.amax(np.abs(z0_vector),axis=0)
    #generating the data
    for i in range(n_exp):
        z_temp = odeint_sym(cartpole, z0_vector[i], time_vector, args=real_args)
        z_real[:,i,:] = z_temp
    print("---------------------------------------------------------------------------------------------------------------------------------------")
    print("Finished the simulation phase")
    print("Saving data...")
    z_real_reshaped = z_real.reshape(z_real.shape[0], -1)
    np.savetxt(csv_file, z_real_reshaped, delimiter=delimiter)  # save data
    print("Data Saved!")
    print("---------------------------------------------------------------------------------------------------------------------------------------")
    x0s = torch.tensor(z_real[0,:,:]).float()
    domain = [-max_values[0], + max_values[0],-max_values[1], + max_values[1]]
    return x0s, torch.tensor(z_real).float(),domain








# import numpy as np


# arr = np.random.rand(5, 4, 3)

# # reshaping the array from 3D
# # matrice to 2D matrice.
# arr_reshaped = arr.reshape(arr.shape[0], -1)

# # saving reshaped array to file.
# np.savetxt("test.txt", arr_reshaped)

# # retrieving data from file.
# loaded_arr = np.loadtxt("test.txt")

# # This loadedArr is a 2D array, therefore
# # we need to convert it to the original
# # array shape.reshaping to get original
# # matrice with original shape.
# load_original_arr = loaded_arr.reshape(
# loaded_arr.shape[0], loaded_arr.shape[1] // arr.shape[2], arr.shape[2])

# # check the shapes:
# print("shape of arr: ", arr.shape)
# print("shape of load_original_arr: ", load_original_arr.shape)

# # check if both arrays are same or not:
# if (load_original_arr == arr).all():
# print("Yes, both the arrays are same")
# else:
# print("No, both the arrays are not same")