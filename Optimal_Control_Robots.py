# # Script for the model identification of a simple non linear pendulum using CT-RENs.
from examples.train_robots import Generate_initial_positions_robots
from datalog.datalog import writing_txt_file
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import time
from math import pi as pi
from torchdiffeq import odeint_adjoint as odeint
import os
from models.FeedbackModels import FeedbackSystem
from src.plots import plot_trajectories, plot_traj_vs_time
from src.loss_functions import loss_ca_daniele,loss_obst_daniele
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=2, help="No. of agents (only 2 implemented).")
    parser.add_argument('--nx', type=int, default= 12, help="No. of states of the model.")
    parser.add_argument('--nq', type=int, default= 6, help="No. of nonlinear feedbacks.")
    parser.add_argument('--n_steps', type=int, default= 50, help="No. of steps used during simulations.")
    parser.add_argument('--t_end', type=float, default= 0.25, help="Dimension of the time window [0, t_end].")
    parser.add_argument('--sigma', type=str, default='relu', help="Activation function of REN_ODE.")
    parser.add_argument('--method', type=str, default='rk4', help="Integration method.")
    parser.add_argument('--seed', type=int, default=2, help="No. of the seed used during simulation.")
    parser.add_argument('--epochs', type=int, default=30, help="(Max) no. of epochs to be used.")
    parser.add_argument('--batch_size', type=int, default=75, help="Size of the batches.")
    parser.add_argument('--alpha', type=float, default= 0.0, help="Contractivity rate.")
    parser.add_argument('--device', type=str, default= 'cuda', help="Choice of the computational device ('cpu' or 'cuda').")
    parser.add_argument('--n_cuda', type=int, default= 0, help="Choice of the Cuda device.")
    parser.add_argument('--learning_rate', type=float, default= 9.e-3,  help="Learning rate.")
    parser.add_argument('--minimum_distance', type=float, default= 0.5,  help="Safe (minimum) distance to preserve between the robots.")
    parser.add_argument('--n_exp', type=int, default=750,  help="No. of initial (random) states of the robots to consider for training AND testing.")
    parser.add_argument('--mode', type=str, default="input_p",  help="Property of the REN_ODE to enforce (contractivity, passivity, ...).")
    parser.add_argument('--robustness_parameter', type=float, default=0.01,  help="hyper-parameter used for robustness in RREN_ODE (e.g., rho). Valid only if 'mode' is not 'c'.")
    args = parser.parse_args()
    
    if (args.device.lower() == 'cuda'):
        device = torch.device('cuda:'+str(args.n_cuda) if torch.cuda.is_available() else 'cpu') 
    else:
        device = torch.device('cpu')
    seed = args.seed
    torch.manual_seed(seed)


    n_agents = args.n_agents             #no. of agents (2 for the 'corridor' experiments, otherwise for the 'swarm' experiment).
    minimum_distance = args.minimum_distance    #safe (minimum) distance to preserve between the robots.

    if (n_agents == 2) :
        experiment = "corridor"
    else:
        experiment = "swarm"
        



    # # Select network parameters
    # n_steps = 50        #time steps used for the evaluation of the cost function. 
    n_steps = args.n_steps     
    t_i =0.0            #initial time 
    # t_end = 40.         #end time
    t_end = args.t_end
    time_vector = np.linspace(t_i, t_end, n_steps)
    # # Select Experiment parameters and file settings
    n_exp = args.n_exp       #no. of initial (random) states of the robots to consider


    # learning_rate = 9e-3
    # epochs = 30
    # batch = 75 
    
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    tol_loss = 1.0e-4   #tolerance of the cost function before stopping it.
    steps_integration = 100
    h = float((t_end - t_i)/(steps_integration-1))

    # #                                     T R A I N I N G    P H A S E 
    # # -------------------------------------------------------------------------------------------------------------------------------------------
    # # Select hyper-parameters of the model
    alpha_x = 100.      #weight on the square of the state x
    alpha_xf = 3000.    #weight on the error on the 
    alpha_u = 0.15      #weight on the squaren of the input
    # alpha_u = 0.2
    alpha_ca = 50.      #weight to avoid collisions between robots
    alpha_obst = 0.08   #weight to avoid collisions with the obstacles (i.e., the mountains)
    # # ---------------------------------
    nq=args.nq               #no. of nonlinear feedback of the REN_ODE
    nx=args.nx               #no. of states of the REN_ODE
    # # ---------------------------------
    # # Choice of the activation function sigma()

    # sigma="relu"
    # sigma="tanh"
    # sigma="sigmoid"
    sigma = args.sigma.lower()
    # # ---------------------------------
    epsilon=5.0e-1
    # # ---------------------------------
    # mode= "c"
    # mode= "rl2"
    # mode = "input_p"
    # mode = "output_p"
    mode = args.mode.lower()
    # constraint = 0.01       #hyper-parameter used for robustness in RREN_ODE (e.g., rho)
    constraint = args.robustness_parameter
    model = FeedbackSystem(nx,nq,sigma,epsilon,mode=mode,sys=experiment,n_agents=n_agents,ni=constraint,gamma=constraint,rho=constraint, device=device, bias=False)
    reference = torch.tensor([2.,2.,0.,0.,-2.,2.,0.,0.], device=device ).unsqueeze(0)
    model.set_reference(reference)      #set the reference point to be used by the controller at each iteration.
    reference_trajectory = reference.repeat(n_steps,1)
    std_x0s = 0.25                      #standard deviation of the initial (random) positions of the robots.

    x0s, training_generator, partition = Generate_initial_positions_robots(nx = nx,batch_size=batch_size,data_size=n_exp, n_agents=n_agents, std =std_x0s, device=device)
    # # TOTAL_DATASETS = [n_steps(time) , n_experiments, nx==n_agents*4]

    training_size = len(partition['train'])
    testing_size = len(partition['test'])
    time_vector = torch.tensor(time_vector,device=device)


    destination_folder=f"./simulations/robots/{experiment}_{mode}/{constraint}/std_{std_x0s}/s_{seed}/"
    if(not(os.path.isdir(destination_folder))):
        os.makedirs(destination_folder)


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
    method =    'rk4'               #Fourth-order Runge-Kutta with 3/8 rule.
    # method =    'explicit_adams'    #Explicit Adams-Bashforth.
    # method =    'implicit_adams'    #Implicit Adams-Bashforth-Moulton.

    #DEFINITION OF THE OPTIMIZER
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, )
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
    optimizer.zero_grad()

    counter = 0
    MSE = nn.MSELoss()
    loss_vector =np.zeros(int((epochs)*np.ceil(n_exp/2/batch_size)))
    epoch=0

    print("\nStarting the Learning phase")
    print("--------------------------------------------------------------------------------------")
    # #                                 T R A I N I N G        P H A S E  
    start = time.time()
    for epoch in range(epochs):
        j = 0
        if epoch >= 5:
            alpha_obst = 1.
        print(f" E P O C H  :  {epoch+1}\n")
        for local_x0s in training_generator:
            #FOR LOOP for the MODEL COMPUTATION
            optimizer.zero_grad()
            actual_batch_size = local_x0s.shape[0] #it might differ if ((datasize/2) % batch) != 0
            
            x_sim = odeint(model, local_x0s , time_vector, method=method,options={'step_size': h})
            U_sim = torch.empty(n_steps,actual_batch_size,model.n_outputs_REN, device=device)
            for nt in range(n_steps):
                U_sim[nt,:,:] = model.input_evaluation(x_sim[nt,:,:])
            
            #calculus of the reference_matrix of the right dimension for the batch size
            # remember, the batch size is constant only if datasize%batch_size == 0
            list_references= []
            for i in range(actual_batch_size):
                list_references.append(reference_trajectory)
            reference_trajectory_batch = torch.stack(list_references, dim=1)
            loss_x = MSE(x_sim[:,:,nx:],reference_trajectory_batch)
            loss_xf = MSE(x_sim[-1-2:,:,nx:],reference_trajectory_batch[-1-2:,:,:])
            loss_u = torch.sum(torch.pow(U_sim,2))
            if n_agents == 2 :
                loss_ca = loss_ca_daniele(x_sim[:,:,nx:],n_agents,min_dist=minimum_distance)
                loss_obst = loss_obst_daniele(x_sim[:,:,nx:],device=device)
            else:
                raise NameError("NOT CONSIDERED MULTIPLE SYSTEMS YET")
            
            loss = alpha_x*loss_x + alpha_u*loss_u + alpha_ca*loss_ca + alpha_obst*loss_obst + alpha_xf*loss_xf
                
            loss.backward()
            optimizer.step()
            model.updateParameters()
            with torch.no_grad():
                loss_vector[counter] = loss
                print(f"Iteration. : {j+1}")
                print(f"Local Loss: {loss:.4f}.\t||\t loss_x: {(alpha_x*loss_x):.2f}.\t||\t loss_u: {(alpha_u*loss_u):.2f}.\t||\t loss_ca: {(alpha_ca*loss_ca):.2f}.\t||\t loss_obst: {(alpha_obst*loss_obst):.2f}\t||\t loss_xf: {(alpha_xf*loss_xf):.2f}")
            counter=counter+1
            j = j + 1
        print("")
    Total_time = time.time()-start
    print("")
    print("")
    print("")
    print(f"Total time required: {Total_time} s")

    # #                                     P L O T T I N G   &   S A V I N G   P H A S E 
    # # --------------------------------------------------------------------------------------------------------------------------------------------  

    with torch.no_grad():
        testing_x0s = x0s[partition["test"]] 
        test_x0 = testing_x0s[0,:].unsqueeze(0)     
        U_log = torch.zeros(n_steps,testing_size,model.n_outputs_REN, device=device)
        x_log = odeint(model, testing_x0s , time_vector, method=method,options={'step_size': h})
        for nt in range(n_steps):
            U_log[nt,:,:] = model.input_evaluation(x_log[nt,:,:])
        list_references= []
        for i in range(testing_x0s.shape[0]):
            list_references.append(reference_trajectory)
        reference_trajectory_testing = torch.stack(list_references, dim=1)
        loss_x_testing = MSE(x_log[:,:,nx:],reference_trajectory_testing)
        loss_xf_testing = MSE(x_log[-1-2:,:,nx:],reference_trajectory_testing[-1-2:,:,:])
        loss_u_testing = torch.sum(torch.pow(U_log,2))
        if n_agents == 2 :
            loss_ca_testing = loss_ca_daniele(x_log[:,:,nx:],n_agents,min_dist=minimum_distance)
            loss_obst_testing = loss_obst_daniele(x_log[:,:,nx:],device=device)
        else:
            raise NameError("MULTIPLE SYSTEMS HAVE NOT BEEN CONSIDERED YET !!!")
        loss_testing = alpha_x*loss_x_testing + alpha_u*loss_u_testing + alpha_ca*loss_ca_testing + alpha_obst*loss_obst_testing + alpha_xf*loss_xf_testing
        print("LOSS TESTING")
        print(f"Total Loss: {loss_testing:.4f}.\t||\t loss_x: {(alpha_x*loss_x_testing):.2f}.\t||\t loss_u: {(alpha_u*loss_u_testing):.2f}.\t||\t loss_ca: {(alpha_ca*loss_ca_testing):.2f}.\t||\t loss_obst: {(alpha_obst*loss_obst_testing):.2f}\t||\t loss_xf: {(alpha_xf*loss_xf_testing):.2f}")
        plot_traj_vs_time(t_end,n_steps, n_agents, x_log[:,0,nx:].squeeze().cpu(), U_log[:,0,:].squeeze().cpu(), save=True,filename=f"Plot_traj_vs_time_TEST_{method}",destination_folder=destination_folder)
        plot_trajectories(x_log[:,0,nx:].squeeze().cpu(),reference.squeeze().cpu(), n_agents, T=n_steps, obst=alpha_obst, save=True,filename=f"Trajectory_TEST_{method}",destination_folder=destination_folder)
        #     #PLOT LOSS VALUE
        loss_vector =  loss_vector[0:counter]  #removing zero values
        plt.figure()
        plt.plot(range(counter), loss_vector[0:counter],linewidth=1, label='loss')
        plt.title("Loss values")
        file = os.path.join(destination_folder,f"Plot_loss_{method}_{sigma}_s{seed}_{n_exp}_exp_{n_steps}_steps.png")
        plt.savefig(file)

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
    Minimum_Distance = {minimum_distance}
    mode = {mode}
    constraint = {constraint}
    ----------------
    Hyper-parameters:
    alpha_x = {alpha_x}
    alpha_u = {alpha_u}
    alpha_ca = {alpha_ca}
    alpha_obst = {alpha_obst}
    nq = {nq}
    ny = {n_agents*2}
    nu = {n_agents*4}
    nx = {nx}
    sigma = {sigma}
    Total training time: {Total_time}
    Final loss:{loss_vector[counter-1]}
    ----------------
    Testing loss:
    loss x testing = {alpha_x*loss_x_testing}
    loss xf testing = {alpha_xf*loss_xf_testing}
    loss u testing = {alpha_u*loss_u_testing}
    loss ca testing = {alpha_ca*loss_ca_testing}
    loss obst testing = {alpha_obst*loss_obst_testing}
    TOTAL loss = {loss_testing}
    """
    writing_txt_file(f"Robot_{experiment}",argument,destination_folder,f"datalog_{mode}_REN_{method}_{sigma}_s_{seed}_{n_exp}_exp_{n_steps}_steps")

    # # Saving the parameters of the model
    path_model = os.path.join(destination_folder,f"Robot_{experiment}_{mode}_REN_{method}_{sigma}_s_{seed}_{n_exp}_exp_{n_steps}_steps.pt")   
    torch.save(model.state_dict(),path_model)