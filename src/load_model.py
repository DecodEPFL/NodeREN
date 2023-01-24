import torch

from src.models import SystemRobots, Controller
from src.plots import plot_trajectories
from src.utils import set_params


def load(sys_model, shocks=False):
    # # # # # # # # Parameters and hyperparameters # # # # # # # #
    if sys_model == "corridor" or sys_model == "robots":
        params = set_params(sys_model)
        min_dist, t_end, n_agents, x0, xbar, linear, _, _, _, _, _, _, n_xi, l, _, _ = params
    else:
        raise ValueError("System model not implemented.")
    # # # # # # # # Define models # # # # # # # #
    sys = SystemRobots(xbar, linear)
    ctl = Controller(sys.f, sys.n, sys.m, n_xi, l)
    # # # # # # # # Define optimizer and parameters # # # # # # # #
    ctl.psi_u.load_state_dict(torch.load("trained_models/" + sys_model + ".pt"))
    ctl.psi_u.eval()
    ctl.psi_u.set_model_param()
    # # # # # # # # Shocks # # # # # # # #
    if sys_model == "corridor" and shocks:
        # Trajectories
        x_log = torch.zeros(t_end, sys.n)
        u_log = torch.zeros(t_end, sys.m)
        w_in = torch.zeros(t_end + 1, sys.n)
        w_in[0, :] = (x0.detach() - sys.xbar)
        u = torch.zeros(sys.m)
        x = sys.xbar
        xi = torch.zeros(ctl.psi_u.n_xi)
        omega = (x, u)
        for t in range(t_end):
            x, _ = sys(t, x, u, w_in[t, :])
            u, xi, omega = ctl(t, x, xi, omega)
            x_log[t, :] = x.detach()
            u_log[t, :] = u.detach()
        # Define the times to do shocks:
        t_s = [12]
        # Define shock values for agent 1:
        x_shock1 = torch.tensor([-1.5, -1, 0, 0.])
        # Define shock values for agent 2:
        x_shock2 = torch.tensor([0.5, -1, 0, 0.])
        # Plot trajectories with shocks
        x_log = torch.zeros(t_end, sys.n)
        u_log = torch.zeros(t_end, sys.m)
        w_in = torch.zeros(t_end + 1, sys.n)
        w_in[0, :] = (x0.detach() - sys.xbar)
        for i in range(len(t_s)):
            w_in[t_s[i], :] = torch.cat([x_shock1, x_shock2])
        u = torch.zeros(sys.m)
        x = sys.xbar
        xi = torch.zeros(ctl.psi_u.n_xi)
        omega = (x, u)
        for t in range(t_end):
            x, _ = sys(t, x, u, w_in[t, :])
            u, xi, omega = ctl(t, x, xi, omega)
            x_log[t, :] = x.detach()
            u_log[t, :] = u.detach()
        plot_trajectories(x_log, xbar, sys.n_agents, text="Shock at time %s" % t_s, T=t_end, obst=True, dots=True)
    return sys, ctl, x0, t_end, min_dist
