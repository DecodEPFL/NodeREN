# # Script for the system identification of a non_linear pendulum using NODE-RENs.
###################################################################################

from examples.train_pendulum import generate_pendulum_dataset
from examples.train_pendulum import generate_uniform_test_pendulum
from models.NODE_REN import NODE_REN
from datalog.datalog import writing_txt_file, makedirs
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import time
from math import pi as pi
from torchdiffeq import odeint_adjoint as odeint
import os

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
    parser.add_argument('--n_steps', type=int, default=300,
                        help="No. of steps used during simulations.")
    parser.add_argument('--t_end', type=float, default=8.,
                        help="Dimension of the total time window [0, t_end]. In the paper is denoted with T_(end)^(test)")
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
                        help="Contractivity rate of NodeREN.")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Choice of the computational device ('cpu' or 'cuda').")
    parser.add_argument('--n_cuda', type=int, default=0,
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
    parser.add_argument('--steps_integration', type=int, default=120,
                        help="Number of integration steps used in fixed-steps methods.")
    parser.add_argument('--n_experiment', type=int, default=0,
                        help="Used to generate irregularly sampled time vectors. Value must be between [0,9]. Default to 0.")
    parser.add_argument('--t_stop_training', type=float, default=1.5,
                        help="Choice of training time interval. In the paper it is denoted with T_end")
    parser.add_argument('--GNODEREN', type=int, default=0,
                        help="If you want to train a GNODEREN (i.e., no constraints). Accepted values: {0,1}")
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
    if (args.method == 'dopri5'):
        destination_folder = f"./simulations/pendulum/Seed_{seed}/{args.method}_{args.atol}"
    else:
        destination_folder = f"./simulations/pendulum/Seed_{seed}/{args.method}_{args.steps_integration}_{args.n_experiment}"
    if (args.GNODEREN == 1):
        destination_folder = f"./simulations/pendulum/Seed_{seed}/GNODEREN_{args.n_experiment}"
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
    # # --------------------------------------------------------
    # # TOTAL_DATASETS = [n_steps(time) , n_experiments, nx==2]
    my_z0s, my_outputs, domain, training_generator, partition = out

    # # Adding noise
    std_noise = 0.05
    my_z0s = my_z0s + std_noise*torch.randn(my_z0s.shape, device=device)
    my_outputs = my_outputs + std_noise * \
        torch.randn(my_outputs.shape, device=device)

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
    if (args.GNODEREN == 1):
        mode = "general"
    else:
        mode = "c"
    model = NODE_REN(nx, ny, nu, nq, sigma, epsilon, mode=mode,
                     device=device, alpha=alpha).to(device)

    print("""NODE-REN:
          
          
       u(nu)┌──────┐   y(ny)
     ──────►│LINEAR├─────►
            │ PART │
         ┌─►│ x(nx)├──┐
    w(nq)│  └──────┘  │v(nq)
         │            │
         │  ┌──────┐  │
         └──┤SIGMA │◄─┘
            │ (*)  │
            └──────┘
          """)
    print(
        f"Chosen dimensions \n nu: {nu} \t||\t nx: {nx} \t||\t nq: {nq} \t||\t ny: {ny}")
    print(f"--> Number of parameters: {count_parameters(model)}")
    print("---------------------------------------------------------------------------------------------------------------------------------------")
    # # CHOICE OF THE INTEGRATION METHOD:
    method = args.method

    # DEFINITION OF THE OPTIMIZER
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    # LEARNING PHASE
    loss = 1.
    counter = 0
    MSE = nn.MSELoss()
    loss_vector = np.zeros((1+epochs)*round(n_exp/batch_size))
    average_NFE_value = NFE_average()
    t_stop = args.t_stop_training
    i = 1
    while (time_vector[i] < t_stop):
        i += 1
    n_index_training = i+1
    time_vector_training = time_vector[0:n_index_training]

    start = time.time()
    print("Starting Training Phase.")
    for epoch in range(epochs):
        # FOR LOOP for the epoch.
        j = 0

        for local_x0s, local_ys in training_generator:
            # FOR LOOP for the MODEL COMPUTATION
            local_ys = torch.permute(local_ys, [1, 0, 2])
            model.nfe = 0
            optimizer.zero_grad()
            # it might differ if ((datasize/2) % batch) != 0
            actual_batch_size = local_x0s.shape[0]
            x0s_enlarged = torch.zeros(actual_batch_size, nx, device=device)
            x0s_enlarged[:, 0:2] = local_x0s
            if (method == 'dopri5'):
                x_sim = odeint(model, x0s_enlarged, time_vector_training, method=method,
                               rtol=args.rtol, atol=args.atol, adjoint_rtol=args.rtol, adjoint_atol=args.atol)
            else:
                x_sim = odeint(model, x0s_enlarged, time_vector_training,
                               method=method, options={'step_size': h})
                # x_sim = odeint(model, x0s_enlarged , time_vector_training, method=method)
            y_sim = torch.empty(
                n_index_training, actual_batch_size, ny, device=device)
            for nt in range(n_index_training):
                y_sim[nt, :, :] = model.output(
                    time_vector_training[nt], x_sim[nt, :, :])
            loss = MSE(y_sim, local_ys[0:n_index_training, :, :])
            nfe_forward = model.nfe
            average_NFE_value.update(nfe_forward)
            model.nfe = 0
            loss.backward()
            optimizer.step()
            model.updateParameters()
            nfe_backward = model.nfe
            model.nfe = 0
            # print(f"NFE-F: {average_NFE_value.average} \t||\t NFE-B: {nfe_backward}")

            with torch.no_grad():
                loss_vector[counter] = loss
                if verbose == 'p':
                    print(
                        f"Epoch #: {epoch+1}.\t||\t Iteration #: {j}.\t||\t Local Loss: {loss:.4f}")
                elif verbose == 'g':
                    total_norm = 0.0
                    for p in model.parameters():
                        param_norm = torch.linalg.norm(
                            p.grad.detach(), dtype=torch.float64)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    print(
                        f"Epoch : {epoch+1}.\t||\t Iteration #: {j}.\t||\t Local Loss: {loss:.4f} \t||\t Gradient Norm: {total_norm:.4E}")
            counter = counter+1
            j = j + 1

        if (abs(loss) < tol_loss):
            print(
                f"The loss has reached a value smaller than the threshold ({tol_loss})")

        with torch.no_grad():
            model.nfe = 0.0
            z_real = my_outputs[:, partition['train'], :]
            x_temp_training = torch.zeros(
                n_steps, training_size, nx, device=device)
            x_temp_training[:, :, 0:2] = z_real[:, :, :]
            x0_training = x_temp_training[0, :, :]
            if (method == 'dopri5'):
                x_training = odeint(model, x0_training, time_vector, method=method, rtol=args.rtol,
                                    atol=args.atol, adjoint_rtol=args.rtol, adjoint_atol=args.atol)
            else:
                x_training = odeint(
                    model, x0_training, time_vector, method=method, options={'step_size': h})
            y_training = torch.empty(n_steps, training_size, ny, device=device)
            for nt in range(n_steps):
                y_training[nt, :, :] = model.output(
                    time_vector[nt], x_training[nt, :, :])
            loss_training = MSE(y_training, z_real)
            # nfe_forward = model.nfe
            # average_NFE_value.update(nfe_forward)
            # print(f"NFE-F: {average_NFE_value.average} \t||\t NFE-B: {nfe_backward}")
            model.nfe = 0
            if verbose != 'n':
                if verbose != 'e':
                    print("---------------------------------------------------------------------------------------------------------------------------------------")
                print(
                    f"\t\t\t\tEpoch : {epoch+1}\t||\t Loss Training: {loss_training:.4f}")
                print(
                    f"\t\t\t\tNFE-F: {average_NFE_value.average} \t||\t NFE-B: {nfe_backward}")
                print("---------------------------------------------------------------------------------------------------------------------------------------")

    Total_time = time.time()-start
    print("")
    print("")
    print("")
    print(f"Finished Training Phase. \nTotal time required: {Total_time} s")

    # #                                     P L O T T I N G   &   S A V I N G   P H A S E
    # # --------------------------------------------------------------------------------------------------------------------------------------------
    time_vector = torch.linspace(0.0, t_end, n_steps, device=device)
    with torch.no_grad():
        method_testing = 'dopri5'
        model.nfe = 0
        # z_testing = my_outputs[:,partition['test'],:]
        z_testing, _ = generate_uniform_test_pendulum(my_z0s[partition['test'], :].cpu().detach(
        ).numpy(), model_parameters, 0.0, t_end=t_end, n_steps=n_steps, seed=seed, device=device)
        z_testing = z_testing + \
            torch.randn(z_testing.shape, device=device)*std_noise
        x0_testing = torch.zeros(testing_size, nx, device=device)
        x0_testing[:, 0:2] = z_testing[0, :, :]
        x_testing = odeint(model, x0_testing, time_vector,
                           method=method_testing)
        y_testing = torch.empty(n_steps, testing_size, ny, device=device)
        for nt in range(n_steps):
            y_testing[nt, :, :] = model.output(
                time_vector[nt], x_testing[nt, :, :])
        loss_testing = MSE(y_testing, z_testing)
        nfe_testing = model.nfe
        print(f"\nLoss_testing: {loss_testing}")
        print(
            f"Final NFE-F average: {average_NFE_value.average} \t||\t NFE-F testing: {nfe_testing}")
        plt.rcParams['font.size'] = 12
        for ind in range(3):
            plt.figure()
            plt.plot(time_vector.cpu().detach().numpy(), z_testing[:, ind, 0].cpu(
            ).detach().numpy(), linewidth=1.5, label=r'$\alpha$')
            plt.plot(time_vector.cpu().detach().numpy(), y_testing[:, ind, 0].cpu(
            ).detach().numpy(), '--', linewidth=1.5, label=r'$\alpha_{NODEREN}$')
            plt.xlabel('Time [s]')
            plt.ylabel(r'$\alpha(t)$ [rad]')
            plt.legend(loc='best')
            # plt.title(f"Final_Plot(#({counter+1})). Time Required: {Total_time:.2f}s. Loss:{loss_testing:.4f}")
            file = os.path.join(destination_folder, f"Plot#{ind}_x1.png")
            plt.savefig(file, dpi=400, bbox_inches='tight')

            plt.figure()
            plt.plot(time_vector.cpu().detach().numpy(), z_testing[:, ind, 1].cpu(
            ).detach().numpy(), linewidth=1.5, label=r'$\dot{\alpha}$')
            plt.plot(time_vector.cpu().detach().numpy(), y_testing[:, ind, 1].cpu(
            ).detach().numpy(), '--', linewidth=1.5, label=r'$\dot{\alpha}_{NODEREN}$')
            plt.xlabel('Time [s]')
            plt.legend(loc='best')
            plt.ylabel(r'$\dot{\alpha}(t)$ [rad/s]')
            file = os.path.join(destination_folder, f"Plot#{ind}_x2.png")
            plt.savefig(file, dpi=400, bbox_inches='tight')

            plt.close('all')

        # PLOT LOSS VALUE
        loss_vector = loss_vector[0:counter]  # removing zero values
        plt.figure()
        plt.plot(range(counter), loss_vector, linewidth=1, label='loss')
        plt.title("Loss values")
        file = os.path.join(destination_folder, f"Plot_loss.png")
        plt.savefig(file)

        argument = f"""Seed:{seed}
        No. experiments: {n_exp}
        No. steps: {n_steps}
        Integration Method: {method}
        t_i = {t_i}
        t_end = {t_end}
        t_stop = {t_stop}
        Actual number of epochs used = {epoch+1}
        Updates done = {counter}
        batch size = {batch_size}
        Steps integration = {steps_integration}
        ----------------
        Hyper-parameters:
        Learning rate: {learning_rate}
        nq = {nq}
        ny = {ny}
        nu = {nu}
        nx = {nx}
        sigma = {sigma}
        Total training time: {Total_time}
        Loss Testing: {loss_testing}
        NFE_testing: {nfe_testing}
        NFE_average: {average_NFE_value.average}
        """

        writing_txt_file("Pendulum", argument, destination_folder, f"Datalog")

    # # Saving the parameters of the model
    path_model = os.path.join(destination_folder, f"Pendulum_C_NODEREN.pt")
    if (args.GNODEREN == 1):
        path_model = os.path.join(destination_folder, f"Pendulum_G_NODEREN.pt")
    torch.save(model.state_dict(), path_model)
