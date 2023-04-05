# Script to use a REN-ODE for binary classification.
# you can choose the type of benchmark with the variable "dataset" and the amount of augmented states with "nf".
import time
from viewers.viewers import viewContour2D, viewTestData
from examples.train_2d_example import generate_train_test
from datalog.datalog import writing_txt_file
from models.NODE_REN import NODE_REN
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
    parser.add_argument('--dataset', type=str,
                        default='double_moons', help="Name of the benchmark.")
    parser.add_argument('--nx', type=int, default=12,
                        help="No. of states of the model.")
    parser.add_argument('--nq', type=int, default=6,
                        help="No. of nonlinear feedbacks.")
    parser.add_argument('--n_layers', type=int, default=40,
                        help="No. of steps used for the evaluation of the output (not used if the integration method uses variable steps).")
    parser.add_argument('--t_end', type=float, default=0.25,
                        help="Dimension of the time window [0, t_end].")
    parser.add_argument('--sigma', type=str, default='relu',
                        help="Activation function of NODE_REN.")
    parser.add_argument('--method', type=str, default='rk4',
                        help="Integration method.")
    parser.add_argument('--seed', type=int, default=10,
                        help="No. of the seed used during simulation.")
    parser.add_argument('--epochs', type=int, default=80,
                        help="(Max) no. of epochs to be used.")
    parser.add_argument('--data_size', type=int, default=12000,
                        help="No. of datapoints simulated (for training AND testing).")
    parser.add_argument('--batch_size', type=int,
                        default=500, help="Size of the batches.")
    parser.add_argument('--alpha', type=float, default=0.0,
                        help="Contractivity rate.")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Choice of the computational device ('cpu' or 'cuda').")
    parser.add_argument('--n_cuda', type=int, default=0,
                        help="Choice of the Cuda device.")
    parser.add_argument('--learning_rate', type=float,
                        default=8.5e-3,  help="Learning rate.")
    args = parser.parse_args()

    if (args.device.lower() == 'cuda'):
        device = torch.device('cuda:'+str(args.n_cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    seed = args.seed
    torch.manual_seed(seed)
    cmap = ListedColormap([[1, 0, 0], [0, 1, 0]])

    dataset, nf = args.dataset, args.nx

    # # Select network parameters
    # No. of steps used for the evaluation of the output (not used if the integration method uses variable steps).
    n_layers = args.n_layers
    t_end = args.t_end
    time_vector = torch.linspace(0., t_end, n_layers, device=device)
    h = float(time_vector[1]-time_vector[0])

    method = args.method  # Integration method
    destination_folder = f"./simulations/classification/{dataset}/Seed_{seed}/xf_{nf}"
    if (not (os.path.isdir(destination_folder))):
        os.makedirs(destination_folder)

    # # Create network

    nq = args.nq  # no. of nonlinear feedbacks
    nx = nf  # no. of states of the model. The model has the same number of states of your system
    ny = 1  # no. of outputs.
    nu = 1  # Negligible (we are considering a free-evolution)

    # # Activation function sigma():
    sigma = args.sigma

    alpha = args.alpha
    epsilon = 8.0e-2
    mode = "c"
    model = NODE_REN(nx, ny, nu, nq, sigma, epsilon, mode=mode,
                     device=device, bias=True, alpha=alpha).to(device)

    # # Select training parameters
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    data_size = args.data_size

    # # Training the network
    out = generate_train_test(
        dataset=dataset, nf=nf, data_size=data_size, batch_size=batch_size, device=device)
    my_data, my_label, domain, training_generator, partition = out
    lossFunc = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_acc = torch.empty(epochs, device=device)
    training_lables = my_label[partition['train']]
    counter_100 = 0
    start_time = time.time()

    for epoch in range(epochs):
        # # FOR LOOP for the epochs.
        j = 0
        # local_samples.shape = [batch_size, nx]
        for local_samples, local_labels in training_generator:
            # # FOR LOOP for the batches
            optimizer.zero_grad()
            if method == 'dopri5':
                xsim = odeint(model, local_samples.squeeze(),
                              time_vector, method=method)
            else:
                xsim = odeint(model, local_samples.squeeze(), time_vector, method=method, options={
                              'step_size': h})    # xsim.shape = [time_istants , n_experiments, nx]

            # ysim.shape = [n_experiments, ny]
            ysim = model.output(time_vector[-1], xsim[-1, :, :])
            loss = lossFunc(ysim, local_labels)
            loss.backward()
            optimizer.step()
            model.updateParameters()
            with torch.no_grad():
                if (j == 0):
                    training_samples = my_data[partition['train']]
                    if method == 'dopri5':
                        x_training = odeint(
                            model, training_samples.squeeze(), time_vector, method=method)
                    else:
                        x_training = odeint(model, training_samples.squeeze(
                        ), time_vector, method=method, options={'step_size': h})
                    y_training = model.output(
                        time_vector[-1], x_training[-1, :, :])
                    training_acc = metrics.accuracy_score(
                        training_lables.cpu(), torch.ge(y_training, 0.0).cpu())
                    loss_acc[epoch] = loss.detach()
                    print(
                        f"Epoch: {epoch} \t||\t Iteration: {j}\t||\t Loss: {loss} \t||\t Acc (training): {training_acc*100:.2f}%")
            j = j + 1
        if (training_acc > 0.9999):
            counter_100 = counter_100 + 1
        else:
            counter_100 = 0

        if counter_100 == 2:
            break
    # # End of the training phase.
    total_time = time.time() - start_time

    # # PLOTS
    plt.figure(1)
    plt.plot(loss_acc.cpu().detach())
    title = os.path.join(destination_folder, f"Loss_plot.png")
    plt.savefig(title)
    plt.figure(2)
    viewContour2D(domain, model, method=method,
                  time_vector=time_vector, device=device)
    viewTestData(partition, my_data.cpu(), my_label.cpu())
    title = os.path.join(destination_folder, f"Contour.png")
    plt.savefig(title)

    with torch.no_grad():
        # # Evaluating the model with the testing dataset.
        testing_samples = my_data[partition['test'], :, :]
        if method == 'dopri5':
            xtest = odeint(model, testing_samples.squeeze(),
                           time_vector, method=method)
        else:
            xtest = odeint(model, testing_samples.squeeze(), time_vector, method=method, options={
                           'step_size': h})  # xtest.shape = [time_istants , n_experiments, nx]
        ytest = model.output(time_vector[-1], xtest[-1, :, :])
        y_predicted = torch.ge(ytest, 0.0).cpu()
        expected_values_testing = my_label[partition['test'], :].cpu()
        test_acc = metrics.accuracy_score(expected_values_testing, y_predicted)
        confusion_matrix = metrics.confusion_matrix(
            expected_values_testing, y_predicted)
        print(f"Accuracy: {test_acc*100.:.2f}%")
        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=[False, True])
        plt.figure()
        cm_display.plot()
        plt.title(f"Accuracy: {test_acc*100.:.2f}%")
        plt.show()
        title = os.path.join(destination_folder, f"Confusion_Matrix.png")
        plt.savefig(title)
        title = os.path.join(destination_folder, f"map2d_points.png")
        plt.figure()
        plt.scatter(testing_samples[:, 0, :].cpu().detach().numpy(
        ), testing_samples[:, -1, :].cpu().detach().numpy(), c=y_predicted.cpu().detach().numpy(), cmap=cmap)
        plt.savefig(title)

        # # Write in a log file the parameters of the experiment.

        argument = f"""Seed:{seed}
        Mode: {mode}
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
        writing_txt_file(
            f"Classification_{dataset}", argument, destination_folder, f"Datalog")
