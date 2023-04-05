import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from src.loss_functions import f_loss_obst
import glob
from PIL import Image
import imageio
def plot_trajectories(x, xbar, n_agents, text="", save=False, filename=None, T=100, obst=False, dots=False,
                      circles=False, axis=True, min_dist=1, f=5, destination_folder="./"):
    fig = plt.figure(f)
    # plt.axes()
    if obst:
        yy, xx = np.meshgrid(np.linspace(-3, 3.5, 120), np.linspace(-3, 3, 100))
        zz = xx * 0
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = f_loss_obst(torch.tensor([xx[i, j], yy[i, j], 0.0, 0.0]), device='cpu')
        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()
        ax = fig.subplots()
        c = ax.pcolormesh(xx, yy, zz, cmap='Greens', vmin=z_min, vmax=z_max)
        # fig.colorbar(c, ax=ax)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(text)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', '#90ee90', '#c20078']
    for i in range(n_agents):
        plt.plot(x[:T+1,4*i].detach(), x[:T+1,4*i+1].detach(), color=colors[i%12], linewidth=1)
        # plt.plot(x[T:,4*i].detach(), x[T:,4*i+1].detach(), color=colors[i%12], linestyle='dotted', linewidth=0.5)
        plt.plot(x[T:,4*i].detach(), x[T:,4*i+1].detach(), color='k', linewidth=0.125, linestyle='dotted')
    for i in range(n_agents):
        plt.plot(x[0,4*i].detach(), x[0,4*i+1].detach(), color=colors[i%12], marker='o', fillstyle='none')
        plt.plot(xbar[4*i].detach(), xbar[4*i+1].detach(), color=colors[i%12], marker='*')
    ax = plt.gca()
    if dots:
        for i in range(n_agents):
            for j in range(T):
                plt.plot(x[j, 4*i].detach(), x[j, 4*i+1].detach(), color=colors[i%12], marker='o')
    if circles:
        for i in range(n_agents):
            r = min_dist/2
            # if obst:
            #     circle = plt.Circle((x[T-1, 4*i].detach(), x[T-1, 4*i+1].detach()), r, color='tab:purple', fill=False)
            # else:
            circle = plt.Circle((x[T, 4*i].detach(), x[T, 4*i+1].detach()), r, color=colors[i%12], alpha=0.5,
                                zorder=10)
            ax.add_patch(circle)
    ax.axes.xaxis.set_visible(axis)
    ax.axes.yaxis.set_visible(axis)
    # TODO: add legend ( solid line: t<T/3 , dotted line> t>T/3, etc )
    if save:
        final_path = os.path.join(destination_folder,f"{filename}_{text}_x_u.pdf")
        plt.savefig(final_path,bbox_inches='tight')
        # plt.savefig('figures/' + filename+'_'+text+'_trajectories.eps', format='eps')
    else:
        plt.show()
    return fig


def plot_traj_vs_time(t_end,n_steps, n_agents, x, u=None, text="", save=False, filename=None,destination_folder="./"):
    # t = torch.linspace(0,t_end-1, t_end)
    t = torch.linspace(0.0,t_end,steps=n_steps)
    if u is not None:
        p = 3
    else:
        p = 2
    plt.figure(figsize=(4*p, 4))
    plt.subplot(1, p, 1)
    plt.grid()
    for i in range(n_agents):
        plt.plot(t, x[:,4*i],label=fr"$x^{i+1}$")
        plt.plot(t, x[:,4*i+1],label=fr"$y^{i+1}$")
    plt.xlabel(r'$t$')
    # plt.ylabel(r'$[m]$')
    plt.title(r'$Positions$')
    plt.legend()
    plt.subplot(1, p, 2)
    plt.grid()
    for i in range(n_agents):
        plt.plot(t, x[:,4*i+2],label=fr"$\dot{'x'}^{i+1}$")
        plt.plot(t, x[:,4*i+3],label=fr"$\dot{'y'}^{i+1}$")
    plt.xlabel(r'$t$')
    # plt.ylabel(r'$[m/s]$')
    plt.title(r'$Velocities$')
    plt.legend()
    plt.suptitle(text)
    if p == 3:
        plt.subplot(1, 3, 3)
        plt.grid()
        for i in range(n_agents):
            plt.plot(t, u[:, 2*i],label=fr"$u_x^{i+1}$")
            plt.plot(t, u[:, 2*i+1],label=fr"$u_y^{i+1}$")
        plt.xlabel(r'$t$')
        # plt.ylabel(r'$[N]$')
        plt.legend()
        plt.title(r'$Inputs$')
    if save:
        final_path = os.path.join(destination_folder,f"{filename}_{text}_x_u.pdf")
        plt.savefig(final_path,bbox_inches='tight')
    else:
        plt.show()



def plot_robots_GIF(x, xbar, n_agents, t_end, min_dist, gif=True, t_plot=100, destination_folder = "./gif", title= "", index_experiment = "", Ts = 1):
    # Extended time
    if(not(os.path.isdir(destination_folder))):
        os.makedirs(destination_folder)
    T = t_plot
    yy, xx = np.meshgrid(np.linspace(-3, 3.5, 120), np.linspace(-3, 3, 100))
    zz = xx * 0
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = f_loss_obst(torch.tensor([xx[i, j], yy[i, j], 0.0, 0.0]))
    z_min, z_max = np.abs(zz).min(), np.abs(zz).max()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', '#90ee90', '#c20078']
    for t_plot in range(0, t_end):
        fig = plt.figure(1, figsize=(6,6))
        ax = plt.gca()
        c = ax.pcolormesh(xx, yy, zz, cmap='Greens', vmin=z_min, vmax=z_max)
        for i in range(n_agents):
            ax.plot(x[:T,4*i].detach(), x[:T,4*i+1].detach(), color=colors[i%12], linewidth=0.25)
            ax.plot(x[T:,4*i].detach(), x[T:,4*i+1].detach(), color=colors[i%12], linestyle='dotted', linewidth=0.125)
        for i in range(n_agents):
            ax.plot(x[0,4*i].detach(), x[0,4*i+1].detach(), color=colors[i%12], marker='*')
            ax.plot(xbar[4*i].detach(), xbar[4*i+1].detach(), color=colors[i%12], marker='o', fillstyle='none')

            r = min_dist / 2
            circle = plt.Circle((x[t_plot,4*i].detach(), x[t_plot,4*i+1].detach()), r, color=colors[i%12],
                                alpha=0.5,
                                zorder=10)
            ax.add_patch(circle)
            ax.axis('equal')
            ax.set(xlim=(-3, +3), ylim=(-3, +3))
            # if t_plot == 1:
            #     xmin, xmax, ymin, ymax = ax.axis()
            # else:
            #     ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        if t_plot > T:
            ax.set_facecolor((0.91, 0.91, 0.91))
        # plt.show()
        plt.title(f"Time: {(t_plot*Ts):.2f}", fontsize=17)
        plt.savefig(f"{destination_folder}/robots_{t_plot}.png")
        plt.close(fig)
    # os.system(f"convert -delay 4 -loop 0 {destination_folder}/robots_*.png {destination_folder}/{title}_{index_experiment}.gif")
    make_gif(frame_folder=destination_folder,title=f"{title}_{index_experiment}", number_figures=t_end, type_of_experiment = "robots")
    os.system(f"rm {destination_folder}/robots_*.png")
#plot_robots(x, clsys.sys.xbar, 1, steps//2, min_dist)




def make_gif(frame_folder, title, extensions = "png", number_figures = 1,type_of_experiment = "robots"):
    frames = []
    slow_start = 20 #frames
    slow_ending = 30
    for i in range(slow_start):
        frames.append(imageio.imread(f"{frame_folder}/{type_of_experiment}_{0}.{extensions}"))
    for i in range(number_figures):
        # frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.{extensions}")]
        frames.append(imageio.imread(f"{frame_folder}/{type_of_experiment}_{i}.{extensions}"))
    for i in range(slow_ending):
        frames.append(imageio.imread(f"{frame_folder}/{type_of_experiment}_{number_figures-1}.{extensions}"))
    final_path = os.path.join(frame_folder,f"{title}.gif")
    imageio.mimsave(final_path, frames, fps=24)
    # frame_one.save(final_path, format="GIF", duration=100, append_images=frames, save_all=True, loop=0)