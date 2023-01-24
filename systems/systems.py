from logging import raiseExceptions
import torch
import torch.nn.functional as F
from torch import matmul as mm
from torch import nn
import numpy as np
import os

from torchdiffeq import odeint_adjoint as odeint


def pendulum(z, t, beta, omega):  # define the ode function
    z1, z2 = z
    dz1 = z2
    dz2 = -omega*omega*np.sin(z1)-beta*z2
    dz = np.array([dz1, dz2])
    return dz