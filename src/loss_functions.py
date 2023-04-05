import torch
import torch.nn.functional as F


def f_loss_states(t, x, sys, Q=None):
    gamma = 1
    if Q is None:
        Q = torch.eye(sys.n)
    xbar = sys.xbar
    dx = x - xbar
    xQx = F.linear(dx, Q) * dx
    return xQx.sum()  # * (gamma**(100-t))


def f_loss_u(t, u):
    loss_u = (u ** 2).sum()
    return loss_u


def f_loss_ca(x, sys, min_dist=0.5):
    min_sec_dist = min_dist + 0.2
    # collision avoidance:
    deltaqx = x[0::4].repeat(sys.n_agents, 1) - x[0::4].repeat(sys.n_agents, 1).transpose(0, 1)
    deltaqy = x[1::4].repeat(sys.n_agents, 1) - x[1::4].repeat(sys.n_agents, 1).transpose(0, 1)
    distance_sq = deltaqx ** 2 + deltaqy ** 2
    mask = torch.logical_not(torch.eye(sys.n // 4))
    loss_ca = (1/(distance_sq + 1e-3) * (distance_sq.detach() < (min_sec_dist ** 2)) * mask).sum()/2
    return loss_ca

def loss_ca(x, n_agents, min_dist=0.5):
    min_sec_dist = min_dist + 0.2
    # collision avoidance:
    eps = 2.e-2
    nt = x.shape[0]
    nexp = x.shape[1]
    loss_ca = 0.0
    for t in range(nt):
        for k in range(nexp):
            deltaqx = x[t,k,0]-x[t,k,4]
            deltaqy = x[t,k,1]-x[t,k,5]
            distance_sq = torch.sqrt(deltaqx ** 2 + deltaqy ** 2)
            if distance_sq <= min_sec_dist:
                loss_ca += 1/((distance_sq+eps))**2
    return loss_ca

def normpdf(q, mu, cov, device):
    d = 2
    mu = mu.view(1, d)
    cov = cov.view(1, d)  # the diagonal of the covariance matrix
    qs = torch.split(q, 2)
    out = torch.tensor(0, device=device)
    for qi in qs:
        # if qi[1]<1.5 and qi[1]>-1.5:
        den = (2*torch.pi)**(0.5*d) * torch.sqrt(torch.prod(cov))
        num = torch.exp((-0.5 * (qi - mu)**2 / cov).sum())
        out = out + num/den
    return out

def loss_obst(x, device= 'cpu'):
    nt = x.shape[0]
    nexp = x.shape[1]
    loss_obst = 0.0
    for t in range(nt):
        for k in range(nexp):
            loss_obst += f_loss_obst(x[t,k,:], device = device)
    return loss_obst

def f_loss_obst(x, device,sys=None):
    # # you can the double colons to slice an array
    # # start:stop:step
    # # ::4    -->  take each multiple of 4
    qx = x[::4].unsqueeze(1)
    qy = x[1::4].unsqueeze(1)
    q = torch.cat((qx,qy), dim=1).view(1,-1).squeeze()
    mu1 = torch.tensor([[-2.5, 0]], device=device)
    mu2 = torch.tensor([[2.5, 0.0]], device=device)
    mu3 = torch.tensor([[-1.5, 0.0]], device=device)
    mu4 = torch.tensor([[1.5, 0.0]], device=device)
    cov = torch.tensor([[0.2, 0.2]], device=device)
    Q1 = normpdf(q, mu=mu1, cov=cov, device=device)
    Q2 = normpdf(q, mu=mu2, cov=cov, device=device)
    Q3 = normpdf(q, mu=mu3, cov=cov, device=device)
    Q4 = normpdf(q, mu=mu4, cov=cov, device=device)

    return (Q1 + Q2 + Q3 + Q4).sum()


def f_loss_side(x):
    qx = x[::4]
    qy = x[1::4]
    side = torch.relu5(qx - 3) + torch.relu(-3 - qx) + torch.relu(qy - 6) + torch.relu(-6 - qy)
    return side.sum()
