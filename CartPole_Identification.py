# # Script for the model identification of a simple non linear pendulum using CT-RENs.
from examples.train_cartpole import generate_train_test_cartpole
from models.REN_ODE import REN_ODE
from datalog.datalog import writing_txt_file
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import time
from math import floor, pi as pi
from torchdiffeq import odeint_adjoint as odeint
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, default= 46, help="No. of states of the model.")
    parser.add_argument('--nq', type=int, default= 15, help="No. of nonlinear feedbacks.")
    parser.add_argument('--n_steps', type=int, default= 90, help="No. of steps used during simulations.")
    parser.add_argument('--t_end', type=float, default= 6.5, help="Dimension of the time window [0, t_end].")
    parser.add_argument('--sigma', type=str, default='tanh', help="Activation function of REN_ODE.")
    parser.add_argument('--method', type=str, default='rk4', help="Integration method.")
    parser.add_argument('--seed', type=int, default=2, help="No. of the seed used during simulation.")
    parser.add_argument('--epochs', type=int, default=160, help="(Max) no. of epochs to be used.")
    parser.add_argument('--batch_size', type=int, default=500, help="Size of the batches.")
    parser.add_argument('--alpha', type=float, default= 0.0, help="Contractivity rate.")
    parser.add_argument('--device', type=str, default= 'cuda', help="Choice of the computational device ('cpu' or 'cuda').")
    parser.add_argument('--n_cuda', type=int, default= 0, help="Choice of the Cuda device.")
    parser.add_argument('--learning_rate', type=float, default= 11.5e-3,  help="Learning rate.")
    parser.add_argument('--n_exp', type=int, default=4000,  help="No. of initial (random) states of the robots to consider for training AND testing.")
    parser.add_argument('--verbose', type=str, default='p',  help="Sets the verbosity level. -'n': none; -'e': once per epoch; -'p': once per iteration; -'g': partial + gradient evaluation (heavy).")
    args = parser.parse_args()
    
    if (args.device.lower() == 'cuda'):
        device = torch.device('cuda:'+str(args.n_cuda) if torch.cuda.is_available() else 'cpu') 
    else:
        device = torch.device('cpu')
    seed = args.seed
    torch.manual_seed(seed)
    # verbose = 'n' # none
    # verbose = 'e' # only epoch
    # verbose = 'p' # partial
    # verbose = 'g' #partial + gradient evaluation. (heavy)
    verbose = args.verbose.lower()
    if not(verbose == 'n' or verbose== 'e' or verbose== 'p' or verbose== 'g' ):
        print("Set wrong value of verbose. The available ones are 'n', 'e', 'p' and 'g'. ")
        print("Verbose was set to 'p'.")
        verbose = 'p'
    # #                                     S I M U L A T I O N    P H A S E 
    # # --------------------------------------------------------------------------------------------------------------------------------------------


    # # Select network parameters
    n_steps = args.n_steps
    t_i =0.0
    t_end = args.t_end
    time_vector = np.linspace(t_i, t_end, n_steps)
    # # Select Experiment parameters and file settings
    real_mp = 2.5
    real_mc = 6.
    real_l = 0.25
    real_beta = 35.
    model_parameters= [real_mp,real_mc,real_l,real_beta]
    n_exp = args.n_exp
    destination_folder=f"./simulations/cartPole/s_{seed}_n_exp_{n_exp}_n_steps_{n_steps}"
    name_file=f"cartPole_s_{seed}_n_exp_{n_exp}_n_steps_{n_steps}"

    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    tol_loss = 1.0e-4
    steps_integration = 235
    h = float((t_end - t_i)/(steps_integration-1))

    out = generate_train_test_cartpole(n_exp,model_parameters, t_i, t_end, n_steps, destination_folder, name_file,seed , data_size=n_exp,  batch_size=batch_size, device = device)
    my_z0s, my_outputs, domain, training_generator, partition = out
    # # TOTAL_DATASETS = [n_steps(time) , n_experiments, nx==2]

    training_size = len(partition['train'])
    testing_size = len(partition['test'])
    time_vector = torch.tensor(time_vector,device=device)
    # #                                               T R A I N I N G    P H A S E 
    # # -------------------------------------------------------------------------------------------------------------------------------------------
    # # Select hyper-parameters of the model
    # nf_enlarged = 42
    # nx=4 + nf_enlarged
    nx = args.nx
    if nx<4:
        nx = 4
    nq=args.nq
    ny=2 
    nu=1    #Negligible for this experiment due to the fact we are considering a free-evolution... 
    # sigma="relu"
    # sigma="tanh"
    # sigma="sigmoid"
    sigma = args.sigma
    epsilon=7.0e-2
    mode="c"
    alpha = args.alpha
    model = REN_ODE(nx, ny , nu , nq ,sigma, epsilon, mode=mode, device=device, bias=True, alpha = alpha).to(device)
    # # CHOICE OF THE INTEGRATION METHOD:
    # # Variable-Step:
    # #--------------------------------------------------------------------------------------------
    # method =    'dopri8'            #Runge-Kutta of order 8 of Dormand-Prince-Shampine.
    # method =    'dopri5'            #Runge-Kutta of order 5 of Dormand-Prince-Shampine [default].
    # method =    'bosh3'             #Runge-Kutta of order 3 of Bogacki-Shampine.
    # method =    'fehlberg2'         #Runge-Kutta-Fehlberg of order 2.
    # method =    'adaptive_heun'     #Runge-Kutta of order 2.

    # # Fixed-step:
    # #--------------------------------------------------------------------------------------------
    # method =    'euler'             #Euler method.
    # method =    'midpoint'          #Midpoint method.
    # method =    'rk4'               #Fourth-order Runge-Kutta with 3/8 rule.
    # method =    'explicit_adams'    #Explicit Adams-Bashforth.
    # method =    'implicit_adams'    #Implicit Adams-Bashforth-Moulton.

    method = args.method
    # device = torch.device('cpu')
    # path_model = os.path.join(destination_folder,f"Pendulum_cREN_{method}_{sigma}_s_{seed}_{n_exp}_exp_{n_steps}_steps.pt")
    # model.load_state_dict(torch.load(path_model, map_location = device))
    # model.updateParameters()


    #DEFINITION OF THE OPTIMIZER
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, )
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    optimizer.zero_grad()
    #LEARNING PHASE
    loss=1.
    counter = 0
    MSE = nn.MSELoss()
    loss_vector =np.zeros((1+epochs)*round(n_exp/batch_size))
    epoch=0
    start = time.time()

    for epoch in range(epochs):
        #FOR LOOP for the epoch.
        j = 0
        for local_x0s, local_ys in training_generator:
            #FOR LOOP for the MODEL COMPUTATION
            local_ys = torch.permute(local_ys,[1,0,2])
            optimizer.zero_grad()
            actual_batch_size = local_x0s.shape[0] #it might differ if ((datasize/2) % batch) != 0
            x0s_enlarged = torch.zeros(actual_batch_size,nx,device=device)
            x0s_enlarged[:,0:4] = local_x0s
            x_sim = odeint(model, x0s_enlarged , time_vector, method=method,options={'step_size': h})
            y_sim = torch.empty(n_steps,actual_batch_size,ny,device=device)
            for nt in range(n_steps):
                y_sim[nt,:,:] = model.output(time_vector[nt],x_sim[nt,:,:])
            # y_sim = torch.bmm(y_sim,weight_matrix)
            # x_temp = torch.bmm(local_ys,weight_matrix)
            # loss = MSE(y_sim,x_temp)
            # loss = MSE(y_sim,local_ys)
            loss = MSE(y_sim,local_ys[:,:,0:ny])
            loss.backward()
            optimizer.step()
            model.updateParameters()
            with torch.no_grad():
                loss_vector[counter]=loss
                if verbose == 'p':
                    print(f"Epoch #: {epoch+1}.\t||\t Iteration #: {j}.\t||\t Local Loss: {loss:.4f}") 
                elif verbose == 'g':
                    total_norm = 0.0
                    for p in model.parameters():
                        param_norm = torch.linalg.norm(p.grad.detach(), dtype=torch.float64)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    print(f"Epoch : {epoch+1}.\t||\t Iteration #: {j}.\t||\t Local Loss: {loss:.4f} \t||\t Gradient Norm: {total_norm:.4E}")
            counter=counter+1
            j = j + 1
            
        with torch.no_grad():
            z_real = my_outputs[:,partition['train'],:]
            x_temp_training = torch.zeros(n_steps,training_size,nx,device=device)
            x_temp_training[:,:,0:4] = z_real[:,:,:]
            x0_training= x_temp_training[0,:,:]
            x_training = odeint(model, x0_training , time_vector, method=method,options={'step_size': h})
            y_training = torch.empty(n_steps,training_size,ny,device=device)
            for nt in range(n_steps):
                y_training[nt,:,:] = model.output(time_vector[nt],x_training[nt,:,:])
            # y_training = torch.bmm(y_training,weight_matrix)
            # x_temp = torch.bmm(z_real,weight_matrix)
            loss_training = MSE(y_training,z_real[:,:,0:ny])
            if verbose != 'n':
                if verbose != 'e':
                    print("---------------------------------------------------------------------------------------------------------------------------------------")
                print(f"\t\t\t\tEpoch : {epoch+1}\t||\t Loss Training: {loss_training:.4f}")
                print("---------------------------------------------------------------------------------------------------------------------------------------")

        if(abs(loss)<tol_loss):
            print(f"The loss has reached a value smaller than the threshold ({loss}).")



    Total_time = time.time()-start
    print("")
    print("")
    print("")
    print(f"Total time required: {Total_time} s")

    # #                                     P L O T T I N G   &   S A V I N G   P H A S E 
    # # --------------------------------------------------------------------------------------------------------------------------------------------  

    with torch.no_grad():
        z_testing = my_outputs[:,partition['test'],:]
        x0_testing = torch.zeros(testing_size,nx,device=device)
        x0_testing[:,0:4] = z_testing[0,:,:]
        x_testing = odeint(model, x0_testing , time_vector, method=method,options={'step_size': h})
        y_testing = torch.empty(n_steps,testing_size,ny,device=device)
        for nt in range(n_steps):
            y_testing[nt,:,:] = model.output(time_vector[nt],x_testing[nt,:,:])
        loss_testing= MSE(y_testing,z_testing[:,:,0:ny])
        print(f"Loss_testing: {loss_testing}\n")

        if ny == 2:
            label_exp = [r'$x$',r'$\theta$']
            label_RENODE = [r'$x_{RENODE}$',r'$\theta_{RENODE}$']
        if ny==4:
            label_exp = [r'$x$',r'$\theta$',r'$\dot{x}$',r'$\dot{\theta}$']
            label_RENODE = [r'$x_{RENODE}$',r'$\theta_{RENODE}$',r'$\dot{x}_{RENODE}$',r'$\dot{\theta}_{RENODE}$']
        for ind in range(3):
            plt.figure()
            plt.grid()
            plt.plot(time_vector.cpu().detach().numpy(), z_testing[:,ind,0:ny].cpu().detach().numpy(), linewidth=1, label=label_exp)
            plt.plot(time_vector.cpu().detach().numpy(), y_testing[:,ind,:].cpu().detach().numpy(), '--', linewidth=1, label=label_RENODE)
            plt.xlabel('Time')
            plt.legend(loc='best')
            plt.title(f"Final_Plot(#({counter+1})). Time Required: {Total_time:.2f}s. Loss:{loss_testing:.4f}")
            file = os.path.join(destination_folder,f"Plot#{ind}_x1_{method}_{sigma}_s{seed}_{n_exp}_exp_{n_steps}_steps.png")
            plt.savefig(file,dpi=400)

            # plt.figure()
            # plt.plot(time_vector, z_testing[:,ind,1].detach().numpy(), 'b', linewidth=1, label='alpha_dot')
            # plt.plot(time_vector, y_testing[:,ind,1].detach().numpy(), 'r--', linewidth=1, label='alpha_dot_REN')
            # plt.xlabel('Time')
            # plt.legend(loc='best')
            # plt.title(f"Final_Plot(#({counter+1})). Time Required: {Total_time:.2f}s. Loss:{loss_testing:.4f}")
            # file = os.path.join(destination_folder,f"Plot#{ind}_x2_{method}_{sigma}_s{seed}_{n_exp}_exp_{n_steps}_steps.png")
            # plt.savefig(file)

        #PLOT LOSS VALUE
        loss_vector =  loss_vector[0:counter]  #removing zero values
        plt.figure()
        plt.plot(range(counter), loss_vector,linewidth=1, label='loss')
        plt.title("Loss values")
        file = os.path.join(destination_folder,f"Plot_loss_{method}_{sigma}_s{seed}_{n_exp}_exp_{n_steps}_steps.png")
        plt.savefig(file)

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {pytorch_total_params}")

        argument = f"""Seed:{seed}
        No. experiments: {n_exp}
        No. steps: {n_steps}
        Integration Method: {method}
        t_i = {t_i}
        t_end = {t_end}
        Actual number of epochs used = {epoch+1}
        Updates done = {counter}
        batch size = {batch_size}
        Steps integration = {steps_integration}
        ----------------
        Parameters of the experiment
            real_mp = {real_mp}
            real_mc = {real_mc}
            real_l = {real_l}
            real_beta = {real_beta}
        ----------------
        Hyper-parameters:
        Total no. of parameters: {pytorch_total_params}
        nq = {nq}
        ny = {ny}
        nu = {nu}
        nx = {nx}
        sigma = {sigma}
        Total training time: {Total_time}
        """

        writing_txt_file("CartPole",argument,destination_folder,f"datalog_{mode}_RENODE_{method}_{sigma}_s_{seed}_{n_exp}_exp_{n_steps}_steps")

    # # Saving the parameters of the model
    path_model = os.path.join(destination_folder,f"CartPole_{mode}RENODE_{method}_{sigma}_s_{seed}_{n_exp}_exp_{n_steps}_steps.pt")
    torch.save(model.state_dict(),path_model)
