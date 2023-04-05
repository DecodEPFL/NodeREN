# # Script for the system identification of a non_linear pendulum using REN-ODEs.
###################################################################################

# from examples.train_pendulum import generate_train_test_pendulum,generate_pendulum_dataset
from examples.train_pendulum import generate_pendulum_dataset
from examples.train_pendulum import generate_uniform_test_pendulum
from models.NODE_REN import NODE_REN
from datalog.datalog import makedirs
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
from math import pi as pi
from torchdiffeq import odeint_adjoint as odeint
import os
import glob

import argparse


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class NFE_average():
    "ARX to calculate the average of the number of function evaluations (NFEs)"

    def __init__(self, mu=0.85):
        self.average = None
        self.mu = mu

    def update(self, value):
        if (self.average == None):
            self.average = value
        else:
            self.average = self.mu * value + (1-self.mu)*self.average
            self.average = np.round(self.average)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, default=4,
                        help="No. of states of the model.")
    parser.add_argument('--nq', type=int, default=5,
                        help="No. of nonlinear feedbacks.")
    parser.add_argument('--n_steps', type=int, default=400,
                        help="No. of steps used during simulations.")
    parser.add_argument('--t_end', type=float, default=8.,
                        help="Dimension of the time window [0, t_end].")
    parser.add_argument('--sigma', type=str, default='tanh',
                        help="Activation function of NODE_REN.")
    parser.add_argument('--method', type=str, default='rk4',
                        help="Integration method.")
    parser.add_argument('--seed', type=int, default=10,
                        help="No. of the seed used during simulation.")
    parser.add_argument('--epochs', type=int, default=30,
                        help="(Max) no. of epochs to be used.")
    parser.add_argument('--batch_size', type=int,
                        default=20, help="Size of the batches.")
    parser.add_argument('--alpha', type=float, default=0.0,
                        help="Contractivity rate.")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Choice of the computational device ('cpu' or 'cuda').")
    parser.add_argument('--n_cuda', type=int, default=3,
                        help="Choice of the Cuda device.")
    parser.add_argument('--learning_rate', type=float,
                        default=10.e-3,  help="Learning rate.")
    parser.add_argument('--n_exp', type=int, default=400,
                        help="No. of initial (random) states of the robots to consider for training AND testing.")
    parser.add_argument('--verbose', type=str, default='p',
                        help="Sets the verbosity level. -'n': none; -'e': once per epoch; -'p': once per iteration; -'g': partial + gradient evaluation (heavy).")
    parser.add_argument('--rtol', type=float, default=1.e-7,
                        help="relative tolerance for 'dopri5'")
    parser.add_argument('--atol', type=float, default=1.e-9,
                        help="absolute tolerance for 'dopri5'")
    parser.add_argument('--steps_integration', type=int, default=110,
                        help="Number of integration steps used in fixed-steps methods.")
    parser.add_argument('--n_experiment', type=int, default=0)
    parser.add_argument('--t_stop_training', type=float, default=3.0,
                        help="Stop time of the simulation during training.")
    args = parser.parse_args()

    if (args.device.lower() == 'cuda'):
        device = torch.device('cuda:'+str(args.n_cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    seed = args.seed
    torch.manual_seed(seed)
    # verbose = 'n' # none
    # verbose = 'e' # only epoch
    # verbose = 'p' # partial
    # verbose = 'g' #partial + gradient evaluation. (heavy)
    verbose = args.verbose.lower()
    if not (verbose == 'n' or verbose == 'e' or verbose == 'p' or verbose == 'g'):
        print("Set wrong value of verbose. The available ones are 'n', 'e', 'p' and 'g'. ")
        print("Verbose was set to 'p'.")
        verbose = 'p'

    # #                                     S I M U L A T I O N    P H A S E
    # # --------------------------------------------------------------------------------------------------------------------------------------------

    # # Select network parameters
    n_steps = args.n_steps
    t_i = 0.0
    t_end = args.t_end
    n_time_vectors = 1
    time_vectors = (t_end-t_i)*torch.rand(int(n_steps)-2, n_time_vectors,
                                          device=device)+t_i*torch.ones(int(n_steps)-2, n_time_vectors, device=device)

    time_vectors = torch.cat((t_i*torch.ones(1, n_time_vectors, device=device),
                             time_vectors, t_end*torch.ones(1, n_time_vectors, device=device)))

    time_vectors, _ = torch.sort(time_vectors, 0)
    time_vectors = torch.unique(time_vectors, dim=0)
    time_vector = time_vectors[:, args.n_experiment]

    real_beta = 1.5
    real_l = 0.5
    model_parameters = [real_beta, real_l]
    n_exp = args.n_exp
    destination_folder = f"./simulations/pendulum/Seed_{seed}/Comparison_GNODEREN__CNODEREN"
    makedirs(destination_folder)

    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    tol_loss = 1.0e-4
    steps_integration = args.steps_integration
    h = float((t_end - t_i)/(steps_integration-1))

    out = generate_pendulum_dataset(n_exp=n_exp,
                                    model_parameters=model_parameters,
                                    time_vector=time_vector,
                                    n_steps=n_steps,
                                    seed=seed,
                                    batch_size=batch_size,
                                    device=device)

    # # TOTAL_DATASETS = [n_steps(time) , n_experiments, nx==2]
    my_z0s, my_outputs, domain, training_generator, partition = out
    std_noise = 0.05
    additive_noise = std_noise*torch.randn(my_outputs.shape, device=device)
    my_outputs = my_outputs + std_noise * \
        torch.randn(my_outputs.shape, device=device)
    my_z0s = my_outputs[0, :, :]
    training_size = len(partition['train'])
    testing_size = len(partition['test'])
    # #                                     T R A I N I N G    P H A S E
    # # -------------------------------------------------------------------------------------------------------------------------------------------
    # # Select hyper-parameters of the model
    nx = args.nx
    nq = args.nq
    ny = 2
    nu = 1
    sigma = args.sigma
    alpha = args.alpha
    epsilon = 5.0e-2
    model_NODEREN = NODE_REN(nx, ny, nu, nq, sigma, epsilon,
                             mode="c", device=device, alpha=alpha).to(device)
    model_G_NODEREN = NODE_REN(
        nx, ny, nu, nq, sigma, epsilon, mode="general", device=device).to(device)

    NODEREN_folder = f"./simulations/pendulum/Seed_{seed}/{args.method}_{args.steps_integration}_{args.n_experiment}"
    G_NODEREN_folder = f"./simulations/pendulum/Seed_{seed}/GNODEREN_{args.n_experiment}"

    path_NODEREN = glob.glob(os.path.join(NODEREN_folder, '*.pt'))[0]
    model_NODEREN.load_state_dict(torch.load(path_NODEREN, map_location='cpu'))
    model_NODEREN.updateParameters()

    path_G_NODEREN = glob.glob(os.path.join(G_NODEREN_folder, '*.pt'))[0]
    model_G_NODEREN.load_state_dict(
        torch.load(path_G_NODEREN, map_location='cpu'))

    # method = args.method
    MSE = nn.MSELoss()
    loss_vector = np.zeros((1+epochs)*round(n_exp/batch_size))
    start = time.time()
    average_NFE_value = NFE_average()
    t_stop = args.t_stop_training
    n_index_training = int(t_stop/t_end*n_steps)
    time_vector_training = time_vector[0:n_index_training]
    print("Starting Testing Phase")
    with torch.no_grad():
        method_testing = 'dopri5'
        z_testing = my_outputs[:, partition['test'], :]
        x0_testing = torch.zeros(testing_size, nx, device=device)
        x0_testing[:, 0:2] = z_testing[0, :, :]

        x_testing_NODEREN = odeint(
            model_NODEREN, x0_testing, time_vector, method=method_testing)
        y_testing_NODEREN = torch.empty(
            n_steps, testing_size, ny, device=device)
        for nt in range(n_steps):
            y_testing_NODEREN[nt, :, :] = model_NODEREN.output(
                time_vector[nt], x_testing_NODEREN[nt, :, :])
        loss_testing_NODEREN = MSE(y_testing_NODEREN, z_testing)

        x_testing_G_NODEREN = odeint(
            model_G_NODEREN, x0_testing, time_vector, method=method_testing)
        y_testing_G_NODEREN = torch.empty(
            n_steps, testing_size, ny, device=device)
        for nt in range(n_steps):
            y_testing_G_NODEREN[nt, :, :] = model_G_NODEREN.output(
                time_vector[nt], x_testing_G_NODEREN[nt, :, :])
        loss_testing_G_NODEREN = MSE(y_testing_G_NODEREN, z_testing)

        plt.rcParams['font.size'] = 15
        for ind in range(3):
            # plt.figure()
            plt.figure("figsize", (6, 4))
            plt.ylim((-2, 2))
            plt.xlim((0, t_end))
            plt.grid()
            plt.plot(t_stop*np.array([1.0, 1.0]), 5.0 *
                     np.array([-1.0, 1.0]), 'k--', linewidth=1.5)
            plt.plot(time_vector.cpu().detach().numpy(), z_testing[:, ind, 0].cpu(
            ).detach().numpy(), color='gray', linewidth=2.0, label='Target')
            plt.plot(time_vector.cpu().detach().numpy(), y_testing_NODEREN[:, ind, 0].cpu(
            ).detach().numpy(), '--', color='orange', linewidth=2.5, label='NodeREN')
            plt.plot(time_vector.cpu().detach().numpy(), y_testing_G_NODEREN[:, ind, 0].cpu(
            ).detach().numpy(), '--', color='tab:blue', linewidth=2.5, label='G-NodeREN')
            plt.xlabel('Time [s]')
            plt.ylabel(r'$\alpha$ [rad]')
            plt.legend(loc='best')
            file = os.path.join(destination_folder, f"Plot#{ind}_x1.pdf")
            plt.savefig(file, bbox_inches='tight')

            plt.figure()
            plt.plot(time_vector.cpu().detach().numpy(), z_testing[:, ind, 1].cpu(
            ).detach().numpy(), linewidth=2.0, label='Target')
            plt.plot(time_vector.cpu().detach().numpy(), y_testing_NODEREN[:, ind, 1].cpu(
            ).detach().numpy(), '--', linewidth=2.5, label='NodeREN')
            plt.plot(time_vector.cpu().detach().numpy(), y_testing_G_NODEREN[:, ind, 1].cpu(
            ).detach().numpy(), '--', linewidth=2.5, label='G-NodeREN')
            plt.xlabel('Time [s]')
            plt.legend(loc='best')
            plt.ylabel(r'$\alpha(t)$ [rad]')
            file = os.path.join(destination_folder, f"Plot#{ind}_x2.pdf")
            plt.savefig(file, bbox_inches='tight')
            plt.close('all')


# #                                     P L O T T I N G   C O N F I D E N C E   I N T E R V A L S
# # ---------------------------------------------------------------------------------------------------------------
    print("Starting Confidence Regions Plots")
    n_traj = 300
    z_testing_confidence, _ = generate_uniform_test_pendulum(my_z0s[partition['test'], :].cpu(
    ).detach().numpy(), model_parameters, 0.0, t_end=t_end, n_steps=n_steps, seed=seed, device=device)
    z_testing_confidence = z_testing_confidence + \
        additive_noise[:, partition['test'], :]
    # z_testing = z_testing + torch.randn(z_testing.shape,device =device)*std_noise
    time_vector = torch.linspace(t_i, t_end, n_steps)
    for k in range(6, 8):
        method = 'dopri5'
        with torch.no_grad():

            z_plot = z_testing_confidence[:, k:k+1, :]
            x0_plot = torch.zeros(n_traj, nx, device=device)
            x0_plot[0, 0:2] = z_plot[0, 0, :]
            for i in range(1, n_traj):
                x0_plot[i, 0:2] = z_plot[0, 0, :] + torch.randn(2,)*0.1
            # ,options={'step_size': h})
            x_plot_NODEREN = odeint(
                model_NODEREN, x0_plot, time_vector, method=method)
            y_plot_NODEREN = torch.empty(
                int(n_steps), n_traj, ny, device=device)
            for nt in range(int(n_steps)):
                y_plot_NODEREN[nt, :, :] = model_NODEREN.output(
                    time_vector[nt], x_plot_NODEREN[nt, :, :])

            # ,options={'step_size': h})
            x_plot_G_NODEREN = odeint(
                model_G_NODEREN, x0_plot, time_vector, method=method)
            y_plot_G_NODEREN = torch.empty(
                int(n_steps), n_traj, ny, device=device)
            for nt in range(int(n_steps)):
                y_plot_G_NODEREN[nt, :, :] = model_G_NODEREN.output(
                    time_vector[nt], x_plot_G_NODEREN[nt, :, :])

        confidence_interval1 = 95
        confidence_interval2 = 80
        confidence_interval3 = 50
        # plt.figure()
        plt.figure("figsize", (6, 4))
        matplotlib.rcParams['text.usetex'] = True

        plt.grid(alpha=0.8)
        for ci in [confidence_interval1, confidence_interval2, confidence_interval3]:
            low_NODEREN = np.percentile(y_plot_NODEREN[:, :, 0].transpose(
                1, 0).cpu().detach().numpy(), 50 - ci / 2, axis=0)
            high_NODEREN = np.percentile(y_plot_NODEREN[:, :, 0].transpose(
                1, 0).cpu().detach().numpy(), 50 + ci / 2, axis=0)
            plt.fill_between(time_vector.cpu().detach().numpy(
            ), low_NODEREN, high_NODEREN, color='orange', alpha=0.2)

            low_G_NODEREN = np.percentile(y_plot_G_NODEREN[:, :, 0].transpose(
                1, 0).cpu().detach().numpy(), 50 - ci / 2, axis=0)
            high_G_NODEREN = np.percentile(y_plot_G_NODEREN[:, :, 0].transpose(
                1, 0).cpu().detach().numpy(), 50 + ci / 2, axis=0)
            plt.fill_between(time_vector.cpu().detach().numpy(
            ), low_G_NODEREN, high_G_NODEREN, color='tab:blue', alpha=0.18)

        plt.plot(t_stop*np.array([1.0, 1.0]), 10.0 *
                 np.array([-1.0, 1.0]), 'k--', label='_nolegend_')
        plt.plot(time_vector.cpu().detach().numpy(), z_plot[:, 0, 0].cpu().detach().numpy(), 'gray', linewidth=1.5,
                 label='Target')
        plt.plot(time_vector.cpu().detach().numpy(), y_plot_NODEREN[:, 0, 0].cpu().detach().numpy(), "--", color='orange', linewidth=1,
                 label='C-NodeREN')
        plt.plot(time_vector.cpu().detach().numpy(), y_plot_G_NODEREN[:, 0, 0].cpu().detach().numpy(), "--", color='tab:blue', linewidth=1,
                 label='G-NodeREN')
        plt.xlabel('Time [s]')
        plt.ylabel(r'${\alpha}(t)$ [rad]')
        plt.legend(loc='best')
        file = os.path.join(destination_folder, f"Plot_confidence_{k}_x1.pdf")
        plt.ylim((-1, 2))
        plt.xlim((0, 8.0))
        plt.savefig(file, bbox_inches='tight')
        plt.close('all')
