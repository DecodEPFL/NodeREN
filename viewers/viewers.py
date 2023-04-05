import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.linalg import null_space
import os
import time
from torchdiffeq import odeint_adjoint as odeint
from data.datasets import _data_extension
import imageio

def viewContour2D(domain, model, input_ch=None,method ="euler",time_vector = None, device="cpu"):
    '''
    Coloured regions in domain represent the prediction of the DNN.
    For 2d datasets.
    '''
    N = 200
    xa = np.linspace(domain[0], domain[1], N)
    ya = np.linspace(domain[2], domain[3], N)
    xv, yv = np.meshgrid(xa, ya)
    y = np.stack([xv.flatten(), yv.flatten()])
    y = np.expand_dims(y.T, axis=2)
    my_data = torch.from_numpy(y).float().to(device)
    nf = model.sys.nx
    if nf != 2:
        my_data = _data_extension(my_data, nf, input_ch, device=device)
    with torch.no_grad():
        h = float(time_vector[1]-time_vector[0])
        if method=='dopri5':
            xsim = odeint(model,my_data.squeeze(),time_vector,method= method)
        else:
            xsim = odeint(model,my_data.squeeze(),time_vector,method= method,options={'step_size': h})    # xsim.shape = [time_istants , n_experiments, nx]
        # xsim = odeint(model,local_samples.squeeze(),time_vector,method="dopri5")
        ysim = model.output(time_vector[-1],xsim[-1,:,:]) 
        labels = torch.ge(ysim, 0.0).int()
    plt.contourf(xa, ya, labels.cpu().view([N, N]), levels=[-0.5, 0.5, 1.5], colors=['#EAB5A0', '#99C4E2'])
    
def viewContour2D_contraction(domain, model, input_ch=None,method ="euler",time_vector = None, device="cpu", number_points = 300):
    '''
    Coloured regions in domain represent the prediction of the DNN.
    For 2d datasets.
    '''
    N = number_points
    xa = np.linspace(domain[0], domain[1], N)
    ya = np.linspace(domain[2], domain[3], N)
    xv, yv = np.meshgrid(xa, ya)
    y = np.stack([xv.flatten(), yv.flatten()])
    y = np.expand_dims(y.T, axis=2)
    my_data = torch.from_numpy(y).float().to(device)
    nf = model.sys.nx
    if nf != 2:
        my_data = _data_extension(my_data, nf, input_ch, device=device)
    with torch.no_grad():
        h = float(time_vector[1]-time_vector[0])
        if method=='dopri5':
            xsim = odeint(model,my_data.squeeze(),time_vector,method= method)
        else:
            xsim = odeint(model,my_data.squeeze(),time_vector,method= method,options={'step_size': h})    # xsim.shape = [time_istants , n_experiments, nx]
        # xsim = odeint(model,local_samples.squeeze(),time_vector,method="dopri5")
        ysim = model.output(time_vector[-1],xsim[-1,:,:]) 
        Sigmoid = torch.nn.Sigmoid()
        labels = torch.ge(Sigmoid(ysim), 0.5).int()
    plt.contourf(xa, ya, labels.cpu().view([N, N]), levels=[-0.5, 0.5, 1.5], colors=['#99C4E2','#EAB5A0'])

def viewTestData(partition, my_data, my_label):
    # Plot test data for 2d datasets.
    test_data_size = len(partition['test'])
    mask0 = (my_label[partition['test'], 0] == 0).view(test_data_size)
    plt.plot(my_data[partition['test'], 0, :].view(test_data_size).masked_select(mask0),
             my_data[partition['test'], -1, :].view(test_data_size).masked_select(mask0), 'r+',
             markersize=2)

    mask1 = (my_label[partition['test'], 0] == 1).view(test_data_size)
    plt.plot(my_data[partition['test'], 0, :].view(test_data_size).masked_select(mask1),
             my_data[partition['test'], -1, :].view(test_data_size).masked_select(mask1), 'b+',
             markersize=2)
    
    
def viewTestData_contraction(partition, my_data, my_label):
    # Plot test data for 2d datasets.
    test_data_size = len(partition['test'])
    mask0 = (my_label[partition['test'], 0] == 0).view(test_data_size)
    plt.plot(my_data[partition['test'], 0, :].view(test_data_size).masked_select(mask0),
             my_data[partition['test'], -1, :].view(test_data_size).masked_select(mask0), 'bx',
             markersize=12)

    mask1 = (my_label[partition['test'], 0] == 1).view(test_data_size)
    plt.plot(my_data[partition['test'], 0, :].view(test_data_size).masked_select(mask1),
             my_data[partition['test'], -1, :].view(test_data_size).masked_select(mask1), 'ro',
             markersize=12)
    
def viewPropagation_circles_contraction(dataset, labels, model,method="euler",time_vector=torch.empty(1),device="cpu"):
    data_size = dataset.shape[0]
    # time_vector= torch.tensor([0.0, 10.0])
    h = time_vector[1]-time_vector[0]
    number_of_samples = 30
    radius = 0.17   
    angle = 2*3.14*torch.linspace(0, 1, number_of_samples)
    for i in range(0, data_size):
        if labels[i] == 0:
            plt.plot(dataset[i, 0], dataset[i, 1], 'bx',markersize = 5)
        if labels[i] == 1:
            plt.plot(dataset[i, 0], dataset[i, 1], 'ro',markersize = 5)
        

        sample = dataset[i]+torch.stack([radius*torch.cos(angle),
                                        radius*torch.sin(angle)]).T
        plt.plot(sample[:, 0], sample[:, 1], '--',color = 'purple')
        if method=='dopri5':
            circles_prop = odeint(model,sample.squeeze().to(device),time_vector,method= method).cpu()
        else:
            circles_prop = odeint(model,sample.squeeze().to(device),time_vector,method= method,options={'step_size': h}).cpu()    # xsim.shape = [time_istants , n_experiments, nx]
        plt.plot(circles_prop[-1, :, 0], circles_prop[-1, :, 1], '--',color = 'orange')

    # plot the classification boundary
    # delta = 0.1
    with torch.no_grad():
        C2 = model.sys.C2
        c1 = C2[0,0]
        c2 = C2[0,1]
        by = model.sys.by
    xa = torch.tensor(-5.)
    xb = torch.tensor(5.)
    ya = (-(c1/c2*xa)-(by-0.5)/c2).cpu()
    yb = (-(c1/c2*xb)-(by-0.5)/c2).cpu()

    x_vector = np.array([xa.detach().numpy(),xb.detach().numpy()])
    y_vector = np.array([ya.squeeze().detach().numpy(),yb.squeeze().detach().numpy()])
    plt.plot(x_vector,y_vector,'g--', linewidth=1.5)
    # plot the trajectory
    if method=='dopri5':
        trajectories = odeint(model,dataset.squeeze().to(device),time_vector,method= method).cpu()
    else:
        trajectories = odeint(model,dataset.squeeze().to(device),time_vector,method= method,options={'step_size': h}).cpu()

    for i in range(data_size):
        if labels[i] == 0:
            plt.plot(trajectories[:,i, 0], trajectories[:,i, 1], 'b--',markersize = 1.5)
        if labels[i] == 1:
            plt.plot(trajectories[:,i, 0], trajectories[:,i, 1], 'r--',markersize = 1.5)
    xv, yv = torch.meshgrid(torch.linspace(-20, 20, 30),
                            torch.linspace(-20, 20, 30))
    # xv, yv = torch.meshgrid(torch.linspace(-8, 8, 25),
                            # torch.linspace(-8, 8, 25))

    y1 = torch.stack([xv.flatten(), yv.flatten()])
    dummy_time = torch.tensor([0.0],device=device)
    with torch.no_grad():
        vector_field = model.sys.forward(dummy_time,y1.T.to(device),torch.zeros(1,model.sys.nu,device=device)).cpu()
    # vector_field = vector_field[-1,:,:]   # so I calculate the evolution over an infinitesimal period
    u = vector_field[:, 0].reshape(xv.size())
    v = vector_field[:, 1].reshape(xv.size())
    print('shape of u',u.shape)
    
    plt.quiver(xv, yv, u, v, color='grey')
    
    
    
    
def viewPropagatedPoints(model, partition, MyData, MyLabel):
    testDataSize = MyLabel[partition['test'], 0].size(0)
    mask0 = (MyLabel[partition['test'], 0] == 0).view(testDataSize)
    YN = model(MyData[partition['test'], :, :]).detach()
    plt.plot(YN[:, 0, :].view(testDataSize).masked_select(mask0),
                YN[:, -1, :].view(testDataSize).masked_select(mask0), 'r+')

    mask1 = (MyLabel[partition['test'], 0] == 1).view(testDataSize)
    plt.plot(YN[:, 0, :].view(testDataSize).masked_select(mask1),
                YN[:, -1, :].view(testDataSize).masked_select(mask1), 'b+')


def view_transf_points(model, partition, MyData, MyLabel, W, mu, nf,YN, n_steps,destination_folder = "./",gif=False):
    # The function calculates a new basis where the W vector coincides with [1,0,0,...,0]
    T = np.zeros((nf, nf))  # the new basis matrix
    B = np.zeros((nf, nf))
    B[0, :] = W
    Z = null_space(B)
    T[:, 0] = W / np.linalg.norm(W)
    T[:, 1:] = Z
    Tinv = np.linalg.inv(T)

    testDataSize = MyLabel[partition['test'], 0].size(0)
    mask0 = (MyLabel[partition['test'], 0] == 0).view(testDataSize)
    mask1 = (MyLabel[partition['test'], 0] == 1).view(testDataSize)
    # YN = model(MyData[partition['test'], :, :]).detach()
    YN_trans = np.zeros_like(YN)
    for i in range(YN_trans.shape[0]):
        YN_trans[i, :, 0] = Tinv.dot(YN[i, :, 0].detach().numpy())
    YN_trans = torch.tensor(YN_trans)

    # Transform the plane
    # support vector
    W_transf = Tinv.dot(W)
    # a point
    p = W / np.linalg.norm(W) * mu  # p is in the classification plane
    p_transf = Tinv.dot(p)

    for axis in range(1, nf):
        plt.figure()

        plt.plot(YN_trans[:, 0, :].view(testDataSize).masked_select(mask0),
                 YN_trans[:, axis, :].view(testDataSize).masked_select(mask0), 'r+')
        plt.plot(YN_trans[:, 0, :].view(testDataSize).masked_select(mask1),
                 YN_trans[:, axis, :].view(testDataSize).masked_select(mask1), 'b+')
        ylim = plt.gca().get_ylim()
        plt.title('Propagated points projected in $x_1$-$x_%s$' % str(axis + 1))
        plt.plot((p[0], p[0]), ylim, 'g--')

    if nf > 2:
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(YN_trans[:, 0, :].view(testDataSize).masked_select(mask0),
                  YN_trans[:, 1, :].view(testDataSize).masked_select(mask0),
                  YN_trans[:, 2, :].view(testDataSize).masked_select(mask0), 'r+')
        ax.plot3D(YN_trans[:, 0, :].view(testDataSize).masked_select(mask1),
                  YN_trans[:, 1, :].view(testDataSize).masked_select(mask1),
                  YN_trans[:, 2, :].view(testDataSize).masked_select(mask1), 'b+')

    if gif:
        for jj in range(n_steps):
            Yj = model(MyData[partition['test'], :, :], end=jj).detach()
            Yj_trans = np.zeros_like(Yj)
            for i in range(Yj_trans.shape[0]):
                Yj_trans[i, :, 0] = Tinv.dot(Yj[i, :, 0].detach().numpy())
            Yj_trans = torch.tensor(Yj_trans)
            axis = 3
            fig = plt.figure(21)
            plt.plot(Yj_trans[:, 0, :].view(testDataSize).masked_select(mask0),
                     Yj_trans[:, axis, :].view(testDataSize).masked_select(mask0), 'r+')
            plt.plot(Yj_trans[:, 0, :].view(testDataSize).masked_select(mask1),
                     Yj_trans[:, axis, :].view(testDataSize).masked_select(mask1), 'b+')
            plt.title('Propagated points at t=%.2f projected in $x_1$-$x_%s$' % (jj*model.h, str(axis + 1)))
            if jj == n_steps-1:
                plt.plot((p[0], p[0]), ylim, 'g--')
            title = os.path.join(destination_folder,f"propagation_{jj}.png")
            plt.savefig(title, format='png',dpi=200)
            plt.close()
        try:
            frames =[]
            for i in range(n_steps):
                frames.append(imageio.imread(f"{destination_folder}/propagation_{i}.png"))
            final_path = os.path.join(destination_folder,f"Final_GIF.gif")
            imageio.mimsave(final_path, frames, fps=24)     
            print("gif created!")
        except:
            print("gif was not created. ImageMagick is needed. See https://imagemagick.org/")
        title = os.path.join(destination_folder,f"propagation_*.png")
        # os.system(f"rm {title}")


def viewEigenvalues(model):
    # Plot eigenvalues of K matrix
    K = model.getK().detach()
    nf = K.size(0)
    level = np.log2(model.N)
    T = model.N*model.h

    x = np.linspace(0, (1-1/model.N)*T, model.N)
    w, _ = np.linalg.eig(K.transpose(0, 2))
    y = np.imag(w.transpose(1, 0))
    for i in range(nf):
        plt.scatter(x, y[i,:])


def viewEigenvalues2(model):
    Kmodel = model.getK().detach()
    J = model.getJ().detach()
    nf = Kmodel.size(0)
    level = np.log2(model.N)
    T = model.N*model.h

    K = Kmodel * 0
    for i in range(Kmodel.size(2)):
        K[:, :, i] = np.matmul(np.matmul(Kmodel[:, :, i], J), Kmodel[:, :, i].transpose(1, 0))
    x = np.linspace(0, (1-1/model.N)*T, model.N)
    w, _ = np.linalg.eig(K.transpose(0, 2))
    y_real = np.real(w.transpose(1, 0))
    y_imag = np.imag(w.transpose(1, 0))
    y_mod = np.sqrt(y_real**2 + y_imag**2)
    plt.figure()
    plt.title("Eigenvalues real part")
    for i in range(nf):
        plt.scatter(x, y_real[i,:])
    plt.figure()
    plt.title("Eigenvalues imaginary part")
    for i in range(nf):
        plt.scatter(x, y_imag[i, :])
    plt.figure()
    plt.title("Eigenvalues modulus")
    for i in range(nf):
        plt.scatter(x, y_mod[i, :])
    plt.plot(x, y_mod.mean(0))



def plot_grad_x_layer(gradients_matrix, colorscale=False, log=True):

    [tot_iters, nf, _, n_layers1] = gradients_matrix.shape
    n_layers = n_layers1 - 1

    if not colorscale:
        plt.figure()
        z = np.linspace(1, n_layers, n_layers)
        legend = []
        for ii in range(1, tot_iters, 100):
            plt.plot(z, np.linalg.norm(gradients_matrix[ii, :, :, :], axis=(0, 1), ord=2)[1:])
            legend.append("Iteration %s" % str(ii))
        for ii in range(1, tot_iters, 1):
            if np.linalg.norm(gradients_matrix[ii, :, :, :], axis=(0, 1), ord=2)[1:].sum() == 0:
                print("zero found at %s" % str(ii))
        plt.xlabel("Layers")
        plt.ylabel(r'$\left\|\frac{\partial y_N}{\partial y_\ell}\right\|$', fontsize=12)
        plt.legend(legend)
    else:
        z = np.linspace(1, n_layers, n_layers)
        fig, ax = plt.subplots()
        n = tot_iters
        # setup the normalization and the colormap
        normalize = mcolors.Normalize(vmin=1, vmax=n)
        colormap = cm.get_cmap('jet', n - 1)
        legend = ['Lower bound']
        ax.plot([1, n_layers], [1, 1], 'k--')
        plt.legend(legend)
        for ii in range(1, n, 1):
            ax.plot(z, np.linalg.norm(gradients_matrix[ii, :, :, :], axis=(0, 1), ord=2)[1:],
                    color=colormap(normalize(ii)),
                    linewidth=0.5)
        plt.xlabel("Layer $\ell$")
        plt.ylabel(r'$\left\|\frac{\partial y_N}{\partial y_\ell}\right\|$', fontsize=12)
        if log:
            plt.yscale('log')
        # setup the colorbar
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        cb = plt.colorbar(scalarmappaple)
        cb.set_label('# iteration')
        plt.tight_layout()


def plot_grad_x_iter(gradients_matrix, colorscale=False, log=True, one_line=True):
    [tot_iters, nf, _, n_layers1] = gradients_matrix.shape
    n_layers = n_layers1 - 1

    if not colorscale:
        plt.figure()
        z = np.linspace(1, tot_iters-1, tot_iters-1)
        legend = []
        for ii in range(1, n_layers):
            plt.plot(z, np.linalg.norm(gradients_matrix[:, :, :, ii], axis=(1, 2), ord=2)[1:])
            legend.append("Layer %s" % str(ii))
        plt.xlabel("Iteration")
        plt.ylabel(r'$\|\|\frac{\partial y_N}{\partial y_\ell}\|\|$', fontsize=12)
        plt.legend(legend)
        return legend
    else:
        x = np.linspace(0, tot_iters - 1, tot_iters)
        fig, ax = plt.subplots()
        n = n_layers
        # setup the normalization and the colormap
        normalize = mcolors.Normalize(vmin=1, vmax=n)
        colormap = cm.get_cmap('jet', n - 1)
        if one_line:
            legend = ['Lower bound']
            ax.plot([0, gradients_matrix.shape[0]], [1, 1], 'k--')
            plt.legend(legend)
        for ii in range(1, n_layers+1):
            j = n_layers-ii
            ax.plot(x, np.linalg.norm(gradients_matrix[:, :, :, j], axis=(1, 2), ord=2), color=colormap(normalize(ii)),
                    linewidth=0.5)
        plt.xlabel("Iterations")
        plt.ylabel(r'$\|\|\frac{\partial y_N}{\partial y_{N-\ell}}\|\|$', fontsize=12)
        if log:
            plt.yscale('log')
        # setup the colorbar
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        cb = plt.colorbar(scalarmappaple)
        cb.set_label('Depth $\ell$')
        plt.tight_layout()
