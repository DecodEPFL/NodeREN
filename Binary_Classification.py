## Script to use a REN-ODE for binary classification.
## you can choose the type of benchmark with the variable "dataset" and the amount of augmented states with "nf".
import time
from viewers.viewers import viewContour2D, viewTestData  # , view_transf_points, plot_grad_x_layer, plot_grad_x_iter
from examples.train_2d_example import generate_train_test
from datalog.datalog import writing_txt_file
from models.REN_ODE import REN_ODE
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from sklearn import metrics
from matplotlib.colors import ListedColormap
import os
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='double_moons', help="Name of the benchmark.")
    parser.add_argument('--nx', type=int, default= 12, help="No. of states of the model.")
    parser.add_argument('--nq', type=int, default= 6, help="No. of nonlinear feedbacks.")
    parser.add_argument('--n_layers', type=int, default= 40, help="No. of steps used for the evaluation of the output (not used if the integration method uses variable steps).")
    parser.add_argument('--t_end', type=float, default= 0.25, help="Dimension of the time window [0, t_end].")
    parser.add_argument('--sigma', type=str, default='relu', help="Activation function of REN_ODE.")
    parser.add_argument('--method', type=str, default='rk4', help="Integration method.")
    parser.add_argument('--seed', type=int, default=2, help="No. of the seed used during simulation.")
    parser.add_argument('--epochs', type=int, default=80, help="(Max) no. of epochs to be used.")
    parser.add_argument('--data_size', type=int, default=12000, help="No. of datapoints simulated (for training AND testing).")
    parser.add_argument('--batch_size', type=int, default=500, help="Size of the batches.")
    parser.add_argument('--alpha', type=float, default= 0.0, help="Contractivity rate.")
    parser.add_argument('--device', type=str, default= 'cuda', help="Choice of the computational device ('cpu' or 'cuda').")
    parser.add_argument('--n_cuda', type=int, default= 0, help="Choice of the Cuda device.")
    parser.add_argument('--learning_rate', type=float, default= 8.5e-3,  help="Learning rate.")
    args = parser.parse_args()
    
    if (args.device.lower() == 'cuda'):
        device = torch.device('cuda:'+str(args.n_cuda) if torch.cuda.is_available() else 'cpu') 
    else:
        device = torch.device('cpu')
    seed = args.seed
    torch.manual_seed(seed)
    cmap = ListedColormap([[1,0,0], [0,1,0]]) 
    # # Select 2D dataset, f function of the NeuralODE (by mode), and number of total states (nf)
    # dataset,  nf = "double_moons", 12
    # dataset,  nf = "swiss_roll", 45
    # dataset,  nf = "circles", 16
    # dataset,  nf = "checker_board", 35
    # dataset,  nf = "letter", 8
    # dataset,  nf = "lines", 24
    dataset, nf = args.dataset, args.nx
    # # Select network parameters
    # n_layers = 40
    n_layers = args.n_layers
    # t_end = .25
    t_end = args.t_end
    time_vector = torch.linspace(0.,t_end,n_layers,device=device)
    h = float(time_vector[1]-time_vector[0])


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
    destination_folder=f"./simulations/classification/{dataset}/s_{seed}/{method}/xf_{nf}"#_n_exp_{data_size}_n_steps_{n_steps}_{sigma}_{method}"
    if(not(os.path.isdir(destination_folder))):
        os.makedirs(destination_folder)

    # # Create network
    # nq=6    #no. of nonlinear feedbacks
    nq = args.nq
    nx=nf   #no. of states of the model. The model has the same number of states of your system 
    ny=1    #no. of outputs.
    nu=1    #Negligible for this experiment due to the fact we are considering a free-evolution... 

    # # Activation function sigma():
    # sigma="tanh" 
    # sigma="relu"
    # sigma = 'sigmoid'
    sigma = args.sigma
    
    # alpha = 0.0
    alpha = args.alpha
    epsilon=8.0e-2
    model = REN_ODE(nx, ny , nu , nq ,sigma, epsilon, device=device, bias=True, alpha=alpha).to(device)

    # # Select training parameters
    # learning_rate = 8.5e-3
    # epochs = 80
    # batch_size = 500
    # data_size = 12000
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    data_size = args.data_size

    # # Training the network
    out = generate_train_test(dataset=dataset, nf=nf, data_size=data_size, batch_size=batch_size, device=device)
    my_data, my_label, domain, training_generator, partition = out
    lossFunc = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_acc = torch.empty(epochs,device = device)
    training_lables = my_label[partition['train']]
    counter_100 = 0
    start_time = time.time()
    time_epoch_vector =np.zeros(epochs)
    for epoch in range(epochs):
        j = 0
        start_time_epoch = time.time()
        for local_samples, local_labels in training_generator:                  #local_samples.shape = [batch_size, nx]
            optimizer.zero_grad()
            if method=='dopri5':
                xsim = odeint(model,local_samples.squeeze(),time_vector,method=method)
            else:
                xsim = odeint(model,local_samples.squeeze(),time_vector,method=method,options={'step_size': h})    # xsim.shape = [time_istants , n_experiments, nx]
            # xsim = odeint(model,local_samples.squeeze(),time_vector,method="dopri5")
            ysim = model.output(time_vector[-1],xsim[-1,:,:]) #ysim.shape = [n_experiments, ny]
            loss = lossFunc(ysim, local_labels)
            loss.backward()
            optimizer.step()
            model.updateParameters()
            with torch.no_grad():
                # if (math.floor(data_size/2/batch_size/2)== j or j == 0):
                if (j == 0):
                    # local_acc = metrics.accuracy_score(local_labels, torch.ge(ysim,0.0))
                    training_samples = my_data[partition['train']]
                    if method=='dopri5':
                        x_training = odeint(model,training_samples.squeeze(),time_vector,method=method)
                    else:
                        x_training = odeint(model,training_samples.squeeze(),time_vector,method=method,options={'step_size': h})
                    y_training = model.output(time_vector[-1],x_training[-1,:,:])
                    training_acc = metrics.accuracy_score(training_lables.cpu(), torch.ge(y_training,0.0).cpu())
                    loss_acc[epoch] = loss.detach()
                    print(f"Epoch: {epoch} \t||\t Iteration: {j}\t||\t Loss: {loss} \t||\t Acc (training): {training_acc*100:.2f}%")
            j = j + 1   
        time_epoch_vector[epoch] = time.time() - start_time_epoch
        if(training_acc>0.9999):
            counter_100 = counter_100 + 1
            # print(f"Counter Incremented. counter_100 = {counter_100}")
        else:
            counter_100 = 0

        if  counter_100 == 6:
            break;    
    total_time = time.time() - start_time


    plt.figure(1)
    plt.plot(loss_acc.cpu().detach())
    # plt.show()
    title = os.path.join(destination_folder,f"Loss_plot.png")
    plt.savefig(title)
    plt.figure(2)
    viewContour2D(domain, model,method=method,time_vector=time_vector, device=device)
    viewTestData(partition, my_data.cpu(), my_label.cpu())
    # plt.show()
    title = os.path.join(destination_folder,f"Contour.png")
    plt.savefig(title)
    plt.figure(3)
    plt.plot(time_epoch_vector)
    title = os.path.join(destination_folder,f"time_epoch_vector.png")
    plt.savefig(title,dpi=300)

    with torch.no_grad():
        testing_samples = my_data[partition['test'], :, :]
        # ytest = torch.zeros(testing_samples.shape[0])
        if method=='dopri5':
            xtest = odeint(model,testing_samples.squeeze(),time_vector,method=method)
        else:
            xtest = odeint(model,testing_samples.squeeze(),time_vector,method=method, options={'step_size': h}) # xtest.shape = [time_istants , n_experiments, nx]
        ytest = model.output(time_vector[-1],xtest[-1,:,:])
        y_predicted =  torch.ge(ytest,0.0).cpu()
        expected_values_testing =my_label[partition['test'], :].cpu()
        test_acc = metrics.accuracy_score(expected_values_testing,y_predicted)
        confusion_matrix = metrics.confusion_matrix(expected_values_testing, y_predicted)
        
        # precision_score = metrics.average_precision_score(expected_values_testing, y_predicted)
        print(f"Accuracy: {test_acc*100.:.2f}%")

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        plt.figure()
        cm_display.plot()
        plt.title(f"Accuracy: {test_acc*100.:.2f}%")
        plt.show()
        title = os.path.join(destination_folder,f"Confusion_Matrix.png")
        plt.savefig(title)
        title = os.path.join(destination_folder,f"map2d_points.png")
        plt.figure()
        plt.scatter(testing_samples[:,0,:].cpu().detach().numpy(), testing_samples[:,-1,:].cpu().detach().numpy(), c=y_predicted.cpu().detach().numpy(), cmap=cmap)
        plt.savefig(title)
        
        argument = f"""Seed:{seed}
        No. experiments: {data_size}
        No. steps: {n_layers}
        Integration Method: {method}
        t_i = {0.0}
        t_end = {t_end}
        Actual number of epochs used = {epoch+1}
        Updates done = {j}
        batch size = {batch_size}
        Steps integration = {h}
        Total_time = {total_time}
        ----------------
        Hyper-parameters:
        nq = {nq}
        ny = {ny}
        nu = {nu}
        nx = {nx}
        sigma = {sigma}
        """

        writing_txt_file(f"Classification_{dataset}",argument,destination_folder,f"datalog_cREN_{method}_{sigma}_s_{seed}_{data_size}_exp_{n_layers}_steps")
