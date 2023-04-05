import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
# import os

# from torchdiffeq import odeint_adjoint as odeint


class _System_contractive(nn.Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, device, bias = False, alpha = 0.0, linear_output=False):
        """Used by the upper class NODE_REN to guarantee contractivity to the model. It should not be used by itself.
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. 
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -alpha (float, optional): Lower bound of the Contraction rate. If alpha is set to 0, the system continues to be contractive, but with a generic (small) rate. Defaults to 0. 
            -linear_output (bool, optional): choose if the output is linear, i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        #Dimensions of Inputs, Outputs, States

        self.nx = nx        #no. internal-states
        self.ny = ny        #no. output
        self.nu = nu        #no. inputs
        self.nq = nq        #no. non-linear states
        self.epsilon = epsilon
        self.device = device
        std = 0.5           #standard deviation used to draw randomly the initial weights of the model.
        #Initialization of the Free Matrices:
        self.Pstar = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        # self.Ptilde = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.Chi = nn.Parameter(torch.randn(nx,nq,device=device)*std)
        #Initialization of the Weights:
        self.Y1 = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.B2 = nn.Parameter(torch.randn(nx,nu,device=device)*std)
        self.D12 = nn.Parameter(torch.randn(nq,nu,device=device)*std)
        self.C2 = nn.Parameter(torch.randn(ny,nx,device=device)*std)
        if (linear_output):
            self.D21 = torch.zeros(ny,nq,device=device)
        else:
            self.D21 = nn.Parameter(torch.randn(ny,nq,device=device)*std)
        self.D22 = nn.Parameter(torch.randn(ny,nu,device=device)*std)
        BIAS = bias
        if(BIAS):
            self.bx= nn.Parameter(torch.randn(nx,1,device=device)*std)
            self.bv= nn.Parameter(torch.randn(nq,1,device=device)*std)
            self.by= nn.Parameter(torch.randn(ny,1,device=device)*std)
        else:
            self.bx= torch.zeros(nx,1,device=device)
            self.bv= torch.zeros(nq,1,device=device)
            self.by= torch.zeros(ny,1,device=device)
        self.X = nn.Parameter(torch.randn(nx+nq,nx+nq,device=device)*std)    # REMEMBER TO CHANGE IT FOR ROBUST SYSTEMS
        #Initialization of the last Parameters:
        self.A = torch.zeros(nx,nx,device=device)
        # self.Y= torch.zeros(nx,nx)
        self.D11 = torch.zeros(nq,nq,device=device) 
        self.C1 = torch.zeros(nq,nx,device=device)
        self.B1 = torch.zeros(nx,nq,device=device)
        self.P = torch.zeros(nx,nx,device=device)
        self.alpha= alpha 
        self.updateParameters()             #Update of: A, B1, C1, D11
        #Choosing the activation function:
        if(sigma == "tanh"):
            self.act = nn.Tanh()
        elif(sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif(sigma == "relu"):
            self.act = nn.ReLU()
        elif(sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def updateParameters(self):
        """Used at the end of each batch training for the update of the constrained matrices.
        """
        P = 0.5*F.linear(self.Pstar,self.Pstar)+self.epsilon*torch.eye(self.nx,device=self.device)
        # P = 0.5*F.linear(self.Pstar,self.Pstar)+self.epsilon*torch.eye(self.nx,device=self.device) + self.Ptilde - self.Ptilde
        self.P = P
        H = F.linear(self.X,self.X) + self.epsilon*torch.eye(self.nx+self.nq,device=self.device)
        #Partition of H in --> [H1 H2;H3 H4]
        h1,h2 = torch.split(H, (self.nx,self.nq), dim =0) # you split the matrices in two big rows
        H1, H2 = torch.split(h1, (self.nx,self.nq), dim=1) # you split each big row in two chunks
        H3, H4 = torch.split(h2, (self.nx,self.nq), dim=1)

        Y= -0.5*(H1+ self.alpha*P + self.Y1-self.Y1.T)
        Lambda = 0.5*torch.diag_embed(torch.diagonal(H4))
        self.A = F.linear(torch.inverse(P),Y.T)
        self.D11 = -F.linear(torch.inverse(Lambda),torch.tril(H4,-1).T)
        self.C1 = F.linear(torch.inverse(Lambda),self.Chi)
        Z = -H2-self.Chi
        self.B1 = F.linear(torch.inverse(P),Z.T)        

    def forward(self,t,xi,u):
        n_initial_states = xi.shape[0]
        By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)# + F.linear(u, self.D12[i,:])
            w = w + F.linear(self.act(v),vec)
        xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(torch.ones(n_initial_states,1,device=self.device),self.bx) + F.linear(u, self.B2)
        # yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_#,yi

    def output(self,xi, u ):
        """Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device)+ F.linear(u, self.D12[0,:])  ).unsqueeze(1)
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
            w = w + F.linear(self.act(v),vec)
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return yt + By

    def calculate_w(self,t,xi,u):
        """Calculates the nonlinear feedback w at time t given the state xi and the input u.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        n_initial_states = xi.shape[0]
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)# + F.linear(u, self.D12[i,:])
            w = w + F.linear(self.act(v),vec)
        return w
    
    def calculate_Vdot(self,delta_x,delta_w,delta_u):
        """Calculates the time-derivative of the storage function V at a given time instant t.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        delta_xdot = F.linear(delta_x, self.A) + F.linear(delta_w, self.B1) + F.linear(torch.ones(1,1),self.bx) + F.linear(delta_u, self.B2)
        Vdot = F.linear(F.linear(delta_xdot,self.P.T),delta_x) + F.linear(F.linear(delta_x,self.P.T),delta_xdot)
        return Vdot
    
    def calculate_s(self,delta_y,delta_u):
        """(Dummy function)
        """
        return 0
    
    
class _System_robust_passive_input(nn.Module):
    def __init__(self, nx, ny, nu, nq,sigma, epsilon, S, Q, R, ni,device, bias = False, alpha=0.0):
        """Used by the upper class NODE_REN to guarantee the model to be (incrementally) input passive (and thus robust). It should not be used by itself.
        
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. 
            -S (torch.tensor): Weight matrix used in the supply rate. Dimensions: nu x ny
            -Q (torch.tensor): Weight matrix used in the supply rate. Dimensions: ny x ny
            -R (torch.tensor): Weight matrix used in the supply rate. Dimensions: nu x nu
            -ni (float): weight coefficient that characterizes the (input passive) supply rate function.
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -alpha (float, optional): Lower bound of the Contraction rate. If alpha is set to 0, the system continues to be contractive, but with a generic (small) rate. Defaults to 0. 
            """
        super().__init__()
        #Dimensions of Inputs, Outputs, States       
        self.nx = nx        #no. internal-states
        self.ny = ny        #no. output
        self.nu = nu        #no. inputs
        self.nq = nq        #no. non-linear states
        self.s = np.max((nu,ny))
        self.epsilon = epsilon
        self.ni = ni
        self.device = device
        self.alpha =alpha
        std = 0.2     #standard deviation used to draw randomly the initial weights of the model.
        
        #For the names of the variables, please refer to the doc file.
        
        #Initialization of the Free Matrices:
        self.Pstar = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.Chi = nn.Parameter(torch.randn(nx,nq,device=device)*std)
        #Initialization of the Weights:
        self.DD12 = nn.Parameter(torch.randn(nq,nu,device=device)*std)
        self.X3 = nn.Parameter(torch.randn(self.s,self.s,device=device)*std)
        self.Y3 = nn.Parameter(torch.randn(self.s,self.s,device=device)*std)
        self.Y1 = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.C2 = nn.Parameter(torch.randn(ny,nx,device=device)*std)
        self.D21 = nn.Parameter(torch.randn(ny,nq,device=device)*std)
        self.Dtilde_1 = nn.Parameter(torch.randn(nu,nu,device=device)*std) 
        BIAS = bias
        if(BIAS):
            self.bx= nn.Parameter(torch.randn(nx,1,device=device)*std)
            self.bv= nn.Parameter(torch.randn(nq,1,device=device)*std)
            self.by= nn.Parameter(torch.randn(ny,1,device=device)*std)
        else:
            self.bx= torch.zeros(nx,1,device=device)
            self.bv= torch.zeros(nq,1,device=device)
            self.by= torch.zeros(ny,1,device=device)
        self.X = nn.Parameter(torch.randn(nx+nq+nu,nx+nq+nu,device=device)*std)    # REMEMBER TO CHANGE IT FOR ROBUST SYSTEMS
        #Initialization of the last Parameters:
        self.A = torch.zeros(nx,nx,device=device)
        self.D11 = torch.zeros(nq,nq,device=device) 
        self.C1 = torch.zeros(nq,nx,device=device)
        self.B1 = torch.zeros(nx,nq,device=device)
        self.D12 = torch.zeros(nq,nu,device=device)
        self.D22 = torch.zeros(ny,nu,device=device)
        self.B2 = torch.zeros(nx,nu,device=device)
        self.R = R
        self.Q = Q
        self.S = S
        self.P = torch.zeros(nx,nx,device=device)
        self.updateParameters()             #Update of: A, B1, C1, D11
        #Choosing the activation function:
        if(sigma == "tanh"):
            self.act = nn.Tanh()
        elif(sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif(sigma == "relu"):
            self.act = nn.ReLU()
        elif(sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def updateParameters(self):
        
        P = 0.5*F.linear(self.Pstar,self.Pstar)+self.epsilon*torch.eye(self.nx,device=self.device)
        self.P = P
        H = F.linear(self.X,self.X) + self.epsilon*torch.eye(self.nx+self.nq+self.ny,device=self.device)
        #Partition of H in --> [H1 H2;H3 H4]
        h1,h2,h3 = torch.split(H, (self.nx,self.nq, self.ny), dim =0) # you split the matrices in two big rows
        H11, H12, H13 = torch.split(h1, (self.nx,self.nq,self.nu), dim=1) # you split each big row in two chunks
        H21, H22, H23 = torch.split(h2, (self.nx,self.nq,self.nu), dim=1) # you split each big row in two chunks
        H31, H32, H33 = torch.split(h3, (self.nx,self.nq,self.nu), dim=1) # you split each big row in two chunks
        V = F.linear(self.C2.T,self.S)-H13
        self.B2 = F.linear(torch.inverse(P),V.T)
        T = F.linear(self.D21.T,self.S)-H23
        Y= -0.5*(H11+ self.alpha*P + self.Y1-self.Y1.T)
        Lambda = 0.5*torch.diag_embed(torch.diagonal(H22))
        self.A = F.linear(torch.inverse(P),Y.T)
        self.D11 = -F.linear(torch.inverse(Lambda),torch.tril(H22,-1).T)
        self.C1 = F.linear(torch.inverse(Lambda),self.Chi)
        Z = -H12-self.Chi
        self.B1 = F.linear(torch.inverse(P),Z.T)
        self.D12 = F.linear(torch.inverse(Lambda),T.T)
        Dtilde =0.5*(H33-self.R+self.Dtilde_1-self.Dtilde_1.T)
        self.D22 = F.linear(torch.inverse(self.S),Dtilde.T)
        
        
    def forward(self,t,xi,u):
        n_initial_states = xi.shape[0]
        By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)# + F.linear(u, self.D12[i,:])
            w = w + F.linear(self.act(v),vec)
        xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(torch.ones(n_initial_states,1,device=self.device),self.bx) + F.linear(u, self.B2)
        # yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_#,yi

    def output(self,xi, u ):
        """Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device)+ F.linear(u, self.D12[0,:])  ).unsqueeze(1)
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
            w = w + F.linear(self.act(v),vec)
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return yt + By
    
    def calculate_w(self,t,xi,u):
        """Calculates the nonlinear feedback w at time t given the state xi and the input u.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        n_initial_states = xi.shape[0]
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)# + F.linear(u, self.D12[i,:])
            w = w + F.linear(self.act(v),vec)
        return w
        
    def calculate_Vdot(self,delta_x,delta_w,delta_u):
        """Calculates the time-derivative of the storage function V at a given time instant t.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        delta_xdot = F.linear(delta_x, self.A) + F.linear(delta_w, self.B1) + F.linear(torch.ones(1,1),self.bx) + F.linear(delta_u, self.B2)
        Vdot = F.linear(F.linear(delta_xdot,self.P.T),delta_x) + F.linear(F.linear(delta_x,self.P.T),delta_xdot)
        return Vdot
    def calculate_s(self,delta_y,delta_u):
        """Calculates the supply rate s(t) at a given time instant t.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        s = 2.*F.linear(delta_u,delta_y) - 2.*self.ni*F.linear(delta_u,delta_u)
        return s

class _System_robust_passive_output(nn.Module):
    def __init__(self, nx, ny, nu, nq,sigma, epsilon, S, Q, R, rho,device, bias = False, alpha = 0.0):
        """Used by the upper class NODE_REN to guarantee the model to be (incrementally) output passive (and thus, robust). It should not be used by itself.
        
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. 
            -S (torch.tensor): Weight matrix used in the supply rate. Dimensions: nu x ny
            -Q (torch.tensor): Weight matrix used in the supply rate. Dimensions: ny x ny
            -R (torch.tensor): Weight matrix used in the supply rate. Dimensions: nu x nu
            -rho (float): weight coefficient that characterizes the (output passive) supply rate function.
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -alpha (float, optional): Lower bound of the Contraction rate. If alpha is set to 0, the system continues to be contractive, but with a generic (small) rate. Defaults to 0. 
            
        """

        super().__init__()
        #Dimensions of Inputs, Outputs, States
        self.nx = nx        #no. internal-states
        self.ny = ny        #no. output
        self.nu = nu        #no. inputs
        self.nq = nq        #no. non-linear states
        self.s = np.max((nu,ny))
        self.epsilon = epsilon
        self.rho = rho
        self.device = device
        self.alpha = alpha
        std = 0.2
        #Initialization of the Free Matrices:
        self.Pstar = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.Chi = nn.Parameter(torch.randn(nx,nq,device=device)*std)
        #Initialization of the Weights:
        self.DD12 = nn.Parameter(torch.randn(nq,nu,device=device)*std)
        self.X3 = nn.Parameter(torch.randn(self.s,self.s,device=device)*std)
        self.Y3 = nn.Parameter(torch.randn(self.s,self.s,device=device)*std)
        self.Y1 = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.C2 = nn.Parameter(torch.randn(ny,nx,device=device)*std)
        self.D21 = nn.Parameter(torch.randn(ny,nq,device=device)*std)
        self.B2 = nn.Parameter(torch.randn(nx,nu,device=device)*std)
        # self.D12 = nn.Parameter(torch.randn(nq,nu,device=device)*std) 
        BIAS = bias
        if(BIAS):
            self.bx= nn.Parameter(torch.randn(nx,1,device=device)*std)
            self.bv= nn.Parameter(torch.randn(nq,1,device=device)*std)
            self.by= nn.Parameter(torch.randn(ny,1,device=device)*std)
        else:
            self.bx= torch.zeros(nx,1,device=device)
            self.bv= torch.zeros(nq,1,device=device)
            self.by= torch.zeros(ny,1,device=device)
        self.X = nn.Parameter(torch.randn(nx+nq,nx+nq,device=device)*std)    # REMEMBER TO CHANGE IT FOR ROBUST SYSTEMS
        #Initialization of the last Parameters:
        self.A = torch.zeros(nx,nx,device=device)
        self.B1 = torch.zeros(nx,nq,device=device)
        self.C1 = torch.zeros(nq,nx,device=device)
        self.D11 = torch.zeros(nq,nq,device=device) 
        self.D12 = torch.zeros(nq,nu,device=device) 
        self.D22 = torch.zeros(ny,nu,device=device)
        # self.B2 = torch.zeros(nx,nu,device=device)
        self.R = R
        self.Q = Q
        self.S = S
        self.Lr = np.sqrt(1./2./self.rho)*torch.eye(nu,device=device)
        self.Lq = np.sqrt(2.*self.rho)*torch.eye(ny,device=device)
        self.P = torch.zeros(nx,nx,device=device)
        self.updateParameters()             #Update of: A, B1, C1, D11
        #Choosing the activation function:
        if(sigma == "tanh"):
            self.act = nn.Tanh()
        elif(sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif(sigma == "relu"):
            self.act = nn.ReLU()
        elif(sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def updateParameters(self):
        P = 0.5*F.linear(self.Pstar,self.Pstar)+self.epsilon*torch.eye(self.nx,device=self.device)
        self.P = P
        M = F.linear(self.X3,self.X3) + self.Y3 - self.Y3.T + self.epsilon*torch.eye(self.s,device=self.device)
        M_tilde = F.linear( torch.eye(self.s,device=self.device)-M , torch.inverse(torch.eye(self.s,device=self.device)+M).T)
        M_tilde = M_tilde[0:self.ny,0:self.nu]
        
        self.D22 = 0.5/self.rho*(M_tilde+ torch.eye(self.ny,device=self.device))
        R_capital = self.D22+self.D22.T -2.*self.rho*F.linear(self.D22.T,self.D22.T)
        V_tilde = -F.linear(P,self.B2.T) +self.C2.T - 2.*self.rho*F.linear(self.C2.T,self.D22.T)
        T_tilde = self.D21.T-self.DD12-2.*self.rho*F.linear(self.D21.T,self.D22.T)
        
        #Finally:
        vec_VT = torch.cat([V_tilde,T_tilde],0)
        vec_C2_D21 = torch.cat([self.C2.T,self.D21.T],0)
        H = F.linear(self.X,self.X) + self.epsilon*torch.eye(self.nx+self.nq,device=self.device) + F.linear(F.linear(vec_VT,torch.inverse(R_capital).T),vec_VT) + 2*self.rho*F.linear(vec_C2_D21,vec_C2_D21) 
        #Partition of H in --> [H1 H2;H3 H4]
        h1,h2 = torch.split(H, (self.nx,self.nq), dim =0) # you split the matrices in two big rows
        H1, H2 = torch.split(h1, (self.nx,self.nq), dim=1) # you split each big row in two chunks
        H3, H4 = torch.split(h2, (self.nx,self.nq), dim=1)

        Y= -0.5*(H1+ self.alpha*P + self.Y1-self.Y1.T)
        Lambda = 0.5*torch.diag_embed(torch.diagonal(H4))
        self.A = F.linear(torch.inverse(P),Y.T)
        self.D11 = -F.linear(torch.inverse(Lambda),torch.tril(H4,-1).T)
        self.C1 = F.linear(torch.inverse(Lambda),self.Chi)
        Z = -H2-self.Chi
        self.B1 = F.linear(torch.inverse(P),Z.T)
        self.D12 = F.linear(torch.inverse(Lambda),self.DD12.T)

        ## Check the LMI for robustness is verified.
        # pr11 = -Y -Y.T
        # pr12 = -Z - self.Chi  
        # pr13 = V_tilde
        # pr21= pr12.T
        # pr22=H4
        # pr23=T_tilde
        # pr31=pr13.T
        # pr32=pr23.T
        # pr33=R_capital
        # temp = torch.cat([self.C2.T,self.D21.T,torch.zeros(self.nu,self.ny)],0)
        # d_temp = -2.*self.rho*F.linear(temp,temp)
        # LMI = torch.cat([
        #     torch.cat([pr11,pr12,pr13],1),
        #     torch.cat([pr21,pr22,pr23],1),
        #     torch.cat([pr31,pr32,pr33],1)],0)+d_temp
        # results = torch.linalg.eigvals(LMI)
        # print(results)
        # print("")
        
    def forward(self,t,xi,u):
        n_initial_states = xi.shape[0]
        #By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)# + F.linear(u, self.D12[i,:])
            w = w + F.linear(self.act(v),vec)
        xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(torch.ones(n_initial_states,1),self.bx) + F.linear(u, self.B2)
        # yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_#,yi

    def output(self,xi, u ):
        """Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device)+ F.linear(u, self.D12[0,:])  ).unsqueeze(1)
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
            w = w + F.linear(self.act(v),vec)
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return yt + By
    
    def calculate_w(self,t,xi,u):
        """Calculates the nonlinear feedback w at time t given the state xi and the input u.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        n_initial_states = xi.shape[0]
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)# + F.linear(u, self.D12[i,:])
            w = w + F.linear(self.act(v),vec)
        return w
    
    def calculate_Vdot(self,delta_x,delta_w,delta_u):
        """Calculates the time-derivative of the storage function V at a given time instant t.
        It is used by the module NODE_REN.calculate_w_Vdot_s_vectors().
        """
        delta_xdot = F.linear(delta_x, self.A) + F.linear(delta_w, self.B1) + F.linear(torch.ones(1,1),self.bx) + F.linear(delta_u, self.B2)
        Vdot = F.linear(F.linear(delta_xdot,self.P.T),delta_x) + F.linear(F.linear(delta_x,self.P.T),delta_xdot)
        return Vdot
    
    def calculate_s(self,delta_y,delta_u):
        s = 2.*F.linear(delta_u,delta_y) - 2.*self.rho*F.linear(delta_y,delta_y)
        return s

class _System_robust_L2_bound(nn.Module):
    def __init__(self, nx, ny, nu, nq,sigma, epsilon, S, Q, R, gamma,device, bias = False, alpha = 0.0):
        """Used by the upper class NODE_REN to guarantee the model to be L2 Lipschitz bounded in its input-output mapping (and thus, robust). It should not be used by itself.
        
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. 
            -S (torch.tensor): Weight matrix used in the supply rate. Dimensions: nu x ny
            -Q (torch.tensor): Weight matrix used in the supply rate. Dimensions: ny x ny
            -R (torch.tensor): Weight matrix used in the supply rate. Dimensions: nu x nu
            -gamma (float): L2 Lipschitz constant.
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -alpha (float, optional): Lower bound of the Contraction rate. If alpha is set to 0, the system continues to be contractive, but with a generic (small) rate. Defaults to 0. 
            """
        super().__init__()
        #Dimensions of Inputs, Outputs, States
        
        self.nx = nx        #no. internal-states
        self.ny = ny        #no. output
        self.nu = nu        #no. inputs
        self.nq = nq        #no. non-linear states
        self.s = np.max((nu,ny))
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device
        self.alpha = alpha
        std = 0.02
        #Initialization of the Free Matrices:
        self.Pstar = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.Chi = nn.Parameter(torch.randn(nx,nq,device=device)*std)
        #Initialization of the Weights:
        self.DD12 = nn.Parameter(torch.randn(nq,nu,device=device)*std)
        self.X3 = nn.Parameter(torch.randn(self.s,self.s,device=device)*std)
        self.Y3 = nn.Parameter(torch.randn(self.s,self.s,device=device)*std)
        self.Y1 = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.B2 = nn.Parameter(torch.randn(nx,nu,device=device)*std)
        self.C2 = nn.Parameter(torch.randn(ny,nx,device=device)*std)
        self.D21 = nn.Parameter(torch.randn(ny,nq,device=device)*std)
        BIAS = bias
        if(BIAS):
            self.bx= nn.Parameter(torch.randn(nx,1,device=device)*std)
            self.bv= nn.Parameter(torch.randn(nq,1,device=device)*std)
            self.by= nn.Parameter(torch.randn(ny,1,device=device)*std)
        else:
            self.bx= torch.zeros(nx,1,device=device)
            self.bv= torch.zeros(nq,1,device=device)
            self.by= torch.zeros(ny,1,device=device)
        self.X = nn.Parameter(torch.randn(nx+nq,nx+nq,device=device)*std)    
        #Initialization of the last Parameters:
        self.A = torch.zeros(nx,nx,device=device)
        self.D11 = torch.zeros(nq,nq,device=device) 
        self.C1 = torch.zeros(nq,nx,device=device)
        self.B1 = torch.zeros(nx,nq,device=device)
        self.D12 = torch.zeros(nq,nu,device=device)
        self.D22 = torch.zeros(ny,nu,device=device)
        
        self.Lq = np.sqrt(1/gamma)*torch.eye(ny,device=device)
        self.Lr = np.sqrt(gamma)*torch.eye(nu,device=device)
        self.R = R
        self.Q = Q
        self.S = S
        
        self.updateParameters()             #Update of: A, B1, C1, D11
        #Choosing the activation function:
        if(sigma == "tanh"):
            self.act = nn.Tanh()
        elif(sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif(sigma == "relu"):
            self.act = nn.ReLU()
        elif(sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()

    def updateParameters(self):
        
        P = 0.5*F.linear(self.Pstar,self.Pstar)+self.epsilon*torch.eye(self.nx,device=self.device)
        M = F.linear(self.X3,self.X3) + self.Y3 - self.Y3.T + self.epsilon*torch.eye(self.s,device=self.device)
        M_tilde = F.linear( torch.eye(self.s,device=self.device)-M , torch.inverse(torch.eye(self.s,device=self.device)+M).T)
        M_tilde = M_tilde[0:self.ny,0:self.nu]
        
        self.D22 = self.gamma*M_tilde
        R_capital = self.R -(1/self.gamma)*F.linear(self.D22.T,self.D22.T)
        V_tilde = -F.linear(P,self.B2.T) -(1/self.gamma)*F.linear(self.C2.T,self.D22.T)
        T_tilde = -self.DD12 -(1/self.gamma)*F.linear(self.D21.T,self.D22.T)
        
        #Finally:
        vec_VT = torch.cat([V_tilde,T_tilde],0)
        vec_C2_D21 = torch.cat([self.C2.T,self.D21.T],0)
        H = F.linear(self.X,self.X) + self.epsilon*torch.eye(self.nx+self.nq,device=self.device) + F.linear(F.linear(vec_VT,torch.inverse(R_capital).T),vec_VT) + np.sqrt(1/self.gamma)*F.linear(vec_C2_D21,vec_C2_D21) 
        #Partition of H in --> [H1 H2;H3 H4]
        h1,h2 = torch.split(H, (self.nx,self.nq), dim =0) # you split the matrices in two big rows
        H1, H2 = torch.split(h1, (self.nx,self.nq), dim=1) # you split each big row in two chunks
        H3, H4 = torch.split(h2, (self.nx,self.nq), dim=1)

        Y= -0.5*(H1+ self.alpha*P + self.Y1-self.Y1.T)
        Lambda = 0.5*torch.diag_embed(torch.diagonal(H4))
        self.A = F.linear(torch.inverse(P),Y.T)
        self.D11 = -F.linear(torch.inverse(Lambda),torch.tril(H4,-1).T)
        self.C1 = F.linear(torch.inverse(Lambda),self.Chi)
        Z = -H2-self.Chi
        self.B1 = F.linear(torch.inverse(P),Z.T)
        self.D12 = F.linear(torch.inverse(Lambda),self.DD12.T)
        
        ## Check the LMI for robustness is verified.
        # pr11 = -Y -Y.T
        # pr12 = -Z - self.Chi  
        # pr13 = -F.linear(P,self.B2.T)
        # pr21= pr12.T
        # pr22=H4
        # pr23=-F.linear(Lambda,self.D12.T)
        # pr31=pr13.T
        # pr32=pr23.T
        # pr33=self.R
        # temp = torch.cat([self.C2.T,self.D21.T,self.D22.T],0)
        # d_temp = -np.sqrt(1/self.gamma)*F.linear(temp,temp)
        # LMI = torch.cat([
        #     torch.cat([pr11,pr12,pr13],1),
        #     torch.cat([pr21,pr22,pr23],1),
        #     torch.cat([pr31,pr32,pr33],1)],0)+d_temp
        # results = torch.linalg.eigvals(LMI)
        # print(results)
        # print("")
        
    def forward(self,t,xi,u):
        n_initial_states = xi.shape[0]
        By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[0,:]) ).unsqueeze(1) 
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)# + F.linear(u, self.D12[i,:])
            w = w + F.linear(self.act(v),vec)
        xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(torch.ones(n_initial_states,1),self.bx) + F.linear(u, self.B2)
        # yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_#,yi

    def output(self,xi, u ):
        """Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        By= F.linear(torch.ones(n_initial_states,1,device=self.device),self.by)
        vec = torch.zeros(self.nq,1,device=self.device)
        vec[0,0] = 1.
        w = torch.zeros(n_initial_states,self.nq,device=self.device)
        v = (F.linear(xi, self.C1[0,:]) + self.bv[0]*torch.ones(n_initial_states,device=self.device)+ F.linear(u, self.D12[0,:])  ).unsqueeze(1)
        w = w + F.linear(self.act(v),vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq,1,device=self.device)
            vec[i,0] = 1.
            v = ( F.linear(xi, self.C1[i,:]) + F.linear(w, self.D11[i,:]) + self.bv[i]*torch.ones(n_initial_states,device=self.device) + F.linear(u, self.D12[i,:]) ).unsqueeze(1)
            w = w + F.linear(self.act(v),vec)
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return yt + By

class _System_general(nn.Module):
    def __init__(self, nx, ny, nu, nq, sigma, epsilon, device, bias = False, linear_output=False):
        """Used by the upper class NODE_REN. It should not be used by itself.
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'.
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. 
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -linear_output (bool, optional): choose if the output is linear, i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        #Dimensions of Inputs, Outputs, States

        self.nx = nx        #no. internal-states
        self.ny = ny        #no. output
        self.nu = nu        #no. inputs
        self.nq = nq        #no. non-linear states
        self.epsilon = epsilon
        self.device = device
        std = .7         #standard deviation used to draw randomly the initial weights of the model.
        #Initialization of the Free Matrices:
        # self.Pstar = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        # self.Chi = nn.Parameter(torch.randn(nx,nq,device=device)*std)
        #Initialization of the Weights:
        # self.Y1 = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.A = nn.Parameter(torch.randn(nx,nx,device=device)*std)
        self.B1 = nn.Parameter(torch.randn(nx,nq,device=device)*std)
        self.B2 = nn.Parameter(torch.randn(nx,nu,device=device)*std)
        self.C1 = nn.Parameter(torch.randn(nq,nx,device=device)*std)
        # self.D11_coefficients = nn.Parameter(torch.randn(nq,device=device)*std) 
        self.D11 = nn.Parameter(torch.randn(nq,nq,device=device)*std) 
        self.D12 = nn.Parameter(torch.randn(nq,nu,device=device)*std)
        self.C2 = nn.Parameter(torch.randn(ny,nx,device=device)*std)
        if (linear_output):
            self.D21 = torch.zeros(ny,nq,device=device)
        else:
            self.D21 = nn.Parameter(torch.randn(ny,nq,device=device)*std)
        self.D22 = nn.Parameter(torch.randn(ny,nu,device=device)*std)
        BIAS = bias
        if(BIAS):
            self.bx= nn.Parameter(torch.randn(nx,1,device=device)*std)
            self.bv= nn.Parameter(torch.randn(nq,1,device=device)*std)
            self.by= nn.Parameter(torch.randn(ny,1,device=device)*std)
        else:
            self.bx= torch.zeros(nx,1,device=device)
            self.bv= torch.zeros(nq,1,device=device)
            self.by= torch.zeros(ny,1,device=device)
        # self.X = nn.Parameter(torch.randn(nx+nq,nx+nq,device=device)*std)    # REMEMBER TO CHANGE IT FOR ROBUST SYSTEMS
        #Initialization of the last Parameters:
        # self.Y= torch.zeros(nx,nx)
        # self.P = torch.zeros(nx,nx,device=device)
        # self.alpha= alpha 
        # self.updateParameters()             #Update of: A, B1, C1, D11
        #Choosing the activation function:
        if(sigma == "tanh"):
            self.act = nn.Tanh()
        elif(sigma == "sigmoid"):
            self.act = nn.Sigmoid()
        elif(sigma == "relu"):
            self.act = nn.ReLU()
        elif(sigma == "identity"):
            self.act = nn.Identity()
        else:
            print("Error. The chosen sigma function is not valid. Tanh() has been applied.")
            self.act = nn.Tanh()     

    def forward(self, t, xi, u):
        n_initial_states = xi.shape[0]
        By = F.linear(torch.ones(n_initial_states, 1, device=self.device), self.by)
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(n_initial_states, self.nq, device=self.device)
        v = (F.linear(xi, self.C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) +
             F.linear(u, self.D12[0,:])).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            v = (F.linear(xi, self.C1[i, :]) + F.linear(w, torch.tril(self.D11,-1)[i, :]) + self.bv[i] * torch.ones(n_initial_states,
                                                                                                     device=self.device)
                 + F.linear(u, self.D12[i, :])).unsqueeze(1)  # + F.linear(u, self.D12[i,:])
            w = w + F.linear(self.act(v), vec)
        xi_ = F.linear(xi, self.A) + F.linear(w, self.B1) + F.linear(
            torch.ones(n_initial_states, 1, device=self.device), self.bx) + F.linear(u, self.B2)
        # yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_  # ,yi
    def updateParameters(self):
        #A general NodeREN does not require any additional step.
        pass
    def output(self, xi, u):
        """Calculates the output yt given the state xi and the input u.
        """
        n_initial_states = xi.shape[0]
        By = F.linear(torch.ones(n_initial_states, 1, device=self.device), self.by)
        vec = torch.zeros(self.nq, 1, device=self.device)
        vec[0, 0] = 1.
        w = torch.zeros(n_initial_states, self.nq, device=self.device)
        v = (F.linear(xi, self.C1[0, :]) + self.bv[0] * torch.ones(n_initial_states, device=self.device) +
             F.linear(u,self.D12[0,:])).unsqueeze(1)
        w = w + F.linear(self.act(v), vec)
        for i in range(1, self.nq):
            vec = torch.zeros(self.nq, 1, device=self.device)
            vec[i, 0] = 1.
            v = (F.linear(xi, self.C1[i, :]) + F.linear(w, torch.tril(self.D11,-1)[i, :]) + self.bv[i] * torch.ones(n_initial_states,
                                                                                                     device=self.device) + F.linear(
                u, self.D12[i, :])).unsqueeze(1)
            w = w + F.linear(self.act(v), vec)
        yt = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return yt + By

class _Controller(nn.Module):
    def __init__(self, nu,device="cpu"):
        """Base class used by NODE_REN to apply an input function u(t) described by the forward() method. It should not be used by itself.

        Args:
            -nu (int): no. of input 
            -device (string, optional): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...). Defaults to "cpu".
        """
        super().__init__()
        self.nu = nu
        self.device =device

    def forward(self, t):
        # # Square input
        # u = (torch.floor(t*3.)%2).unsqueeze(0)*torch.ones(1,self.nu, device = self.device)
        
        # # White noise input
        # u = self.noise[int(torch.floor(t*10))].unsqueeze(0)
        
        # # Unitary Step
        # u = torch.ones(1,self.nu,device=self.device)

        # # Free Evolution
        u = torch.zeros(1,self.nu,device=self.device)
        
        return u


class NODE_REN(nn.Module):
    def __init__(self, nx = 5, ny = 5, nu = 5, nq = 5, sigma = "tanh", epsilon = 1.0e-2, mode = "c", gamma = 1., device = "cpu", bias = False, ni = 1., rho = 1., alpha = 0.0, linear_output=False):
        """Base class for Neural Ordinary Differential Equation Recurrent Equilbrium Networks (NODE_RENs).
        
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'. Defaults to "tanh".
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. Defaults to 1.0e-2.
            -mode (str, optional): Property to ensure. Possible options: 'c'= contractive model, 'rl2'=L2 lipschitz-bounded, 'input_p'=input passive model, 'output_p'=output_passive model.
            -gamma (float, optional): If the model is L2 lipschitz bounded (i.e., mode == 'c'), gamma is the L2 Lipschitz constant. Defaults to 1.
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -ni  (float, optional): If the model is input passive (i.e., mode == 'input_p') , ni is the weight coefficient that characterizes the (input passive) supply rate function.
            -rho (float, optional): If the model is output passive (i.e., mode == 'output_p'), rho is the weight coefficient that characterizes the (output passive) supply rate function.
            -alpha (float, optional): Lower bound of the Contraction rate. If alpha is set to 0, the system continues to be contractive, but with a generic (small) rate. Defaults to 0. 
            -linear_output (bool, optional): choose if the output is linear, i.e., choose to force (or not) the matrix D21 to be null. Defaults to False.
        """
        super().__init__()
        self.ctl = _Controller(nu,device)
        self.mode = mode.lower()
        self.nfe = 0
        if (self.mode == "c"):
            self.sys = _System_contractive(nx, ny, nu, nq,sigma, epsilon, device=device,bias=bias, linear_output=linear_output,alpha=alpha)
        elif (self.mode == "general"):
            self.sys = _System_general(nx, ny, nu, nq,sigma, epsilon, device=device,bias=bias, linear_output=linear_output)
        elif(self.mode == "rl2"):
            Q = -(1./gamma)*torch.eye(ny,device=device)
            R = (gamma)*torch.eye(nu,device=device)
            S = torch.zeros(nu,ny,device=device)
            self.sys = _System_robust_L2_bound(nx, ny, nu, nq,sigma, epsilon,S=S,Q=Q,R=R,gamma=gamma,device=device, bias = bias,alpha=alpha)
        elif(self.mode == "input_p"):
            if (ny != nu):
                raise NameError("u and y have different dimensions, so you cannot have passivity!")
            Q = torch.zeros(ny,device=device)
            R = -2.*ni*torch.eye(nu,device=device)
            S = torch.eye(ny,device=device)
            self.sys = _System_robust_passive_input(nx, ny, nu, nq,sigma, epsilon,S=S,Q=Q,R=R,ni=ni,device=device, bias = bias,alpha=alpha)
        elif(self.mode == "output_p"):
            if (ny != nu):
                raise NameError("u and y have different dimensions, so you cannot have passivity!")
            Q = -2.*rho*torch.eye(ny,device=device)
            R = torch.zeros(nu,device=device)
            S = torch.eye(ny,device=device)
            self.sys = _System_robust_passive_output(nx, ny, nu, nq,sigma, epsilon,S=S,Q=Q,R=R,rho=rho,device=device, bias = bias,alpha=alpha)
        else:
            raise NameError("The inserted mode is not valid. Please write 'c', 'rl2', 'input_p' or 'output_p'. :(")

    def updateParameters(self):
        self.sys.updateParameters()

    def forward(self, t, x):
        self.nfe += 1
        u = self.ctl(t)
        xdot = self.sys(t, x, u)
        # xdot = self.sys(t,x)
        return xdot

    def output(self,t,x):
        u = self.ctl(t)
        yt = self.sys.output(x,u)
        return yt
    @property
    def nfe(self):
        return self._nfe
    @nfe.setter
    def nfe(self,value):
        self._nfe = value
    def change_controller(self,nu,number):
        if number == 2:
            self.ctl = _Controller_2(nu)
        if number == 3:
            self.ctl = _Controller_3(nu)
    
    def calculate_w_Vdot_s_vectors(self,x,u,y,time_vector):
        """Calculates the vectors w, Vdot and s for the given time_vector, where w is the non_linear feedback, Vdot is the time-derivative of the storage function V(t) and s the supply rate function.
        
        Args:
            -x (torch.tensor): State vector. Dimension: n_steps x nx
            -u (torch.tensor): Input vector. Dimension: n_steps x nu
            -y (torch.tensor): Output vector. Dimension: n_steps x ny
            -time_vector (torch.tensor): Given a sampling period Ts, time_vector is a n_steps equally spaced vector. Dimension: n_steps x 1
        Returns:
            w, Vdot, s (all vectors).
        """
        n_steps = time_vector.shape[0]
        w=torch.zeros(n_steps,self.sys.nq)
        Vdot = torch.zeros(n_steps,1)
        s = torch.zeros(n_steps,1)
        for ni in range(n_steps):
            ti = time_vector[ni]
            xi = x[ni,:]
            ui = u[ni,:]
            yi = y[ni,:]
            w[ni,:]=self.sys.calculate_w(ti,xi,ui)
            Vdot[ni,:]=self.sys.calculate_Vdot(xi,w[ni,:],ui)
            s[ni,:]=self.sys.calculate_s(yi,ui)
        return w, Vdot, s  
    
    def calculate_alpha_max(self):
        # A = self.sys.A
        P = self.sys.P
        Y = self.sys.Y_alpha_0
        A = F.linear(torch.inverse(P),Y.T)
        C1 = self.sys.C1
        D11 = self.sys.D11
        Lambda = self.sys.Lambda
        B1 = self.sys.B1 
        W = 2*Lambda - F.linear(Lambda,D11.T) - F.linear(D11.T,Lambda.T)
        C1TLambda_PB1 = F.linear(C1.T,Lambda.T)+F.linear(P,B1.T)
        Phi =  -F.linear(A.T,P.T) - F.linear(P,A.T) - F.linear(F.linear(C1TLambda_PB1,torch.inverse(W).T),C1TLambda_PB1)
        eigs_P = torch.linalg.eigvals(P)
        eigs_Phi = torch.linalg.eigvals(Phi)
        
        lMAX_P =torch.max(torch.real(eigs_P))
        lmin_Phi = torch.min(torch.real(eigs_Phi))
        # print(f"lMAX_P: {lMAX_P}.  lmin_Phi: {lmin_Phi}.")
        alpha_max = lmin_Phi/lMAX_P
        prova = Phi-alpha_max*P
        is_it_true = torch.min(torch.real(torch.linalg.eigvals(prova)))
        if (is_it_true <= 0.0):
            raise NameError("Houston, we have a problem!")
        return alpha_max
    
    def calculate_beta(self):
        # A = self.sys.A
        P = self.sys.P
        # Y = self.sys.Y
        A = self.sys.A
        C1 = self.sys.C1
        D11 = self.sys.D11
        Lambda = self.sys.Lambda
        B1 = self.sys.B1 
        W = 2*Lambda - F.linear(Lambda,D11.T) - F.linear(D11.T,Lambda.T)
        C1TLambda_PB1 = F.linear(C1.T,Lambda.T)+F.linear(P,B1.T)
        Phi =  -F.linear(A.T,P.T) - F.linear(P,A.T) - F.linear(F.linear(C1TLambda_PB1,torch.inverse(W).T),C1TLambda_PB1)- self.sys.alpha*P
        eigs_P = torch.linalg.eigvals(P)
        eigs_Phi = torch.linalg.eigvals(Phi)
        
        lMAX_P =torch.max(torch.real(eigs_P))
        lmin_Phi = torch.min(torch.real(eigs_Phi))
        # print(f"lMAX_P: {lMAX_P}.  lmin_Phi: {lmin_Phi}.")
        beta = lmin_Phi/lMAX_P
        # prova = Phi-alpha_max*P
        # is_it_true = torch.min(torch.real(torch.linalg.eigvals(prova)))
        # if (is_it_true <= 0.0):
        #     raise NameError("Houston, we have a problem!")
        return beta

class _Controller_2(nn.Module):
    def __init__(self, nu):
        super().__init__()
        self.nu = nu

    def forward(self, t):
        # # Square input
        # u = (torch.floor(t)%2).unsqueeze(0)*torch.ones(1,self.nu)
        u = -2.*np.exp(-0.2*t)*np.sin(0.5*np.pi*t+np.pi/3.)*torch.ones(1,self.nu)
        
        # # White noise input
        # u = self.noise[int(torch.floor(t*10))].unsqueeze(0)

        # # Step Evolution
        # u = 10.*torch.ones(1,self.nu)
        return u


class _Controller_3(nn.Module):
    def __init__(self, nu):
        super().__init__()
        self.nu = nu

    def forward(self, t):
        # # Square input
        # u = (torch.floor(t)%2).unsqueeze(0)*torch.ones(1,self.nu)
        u = 3.*np.exp(-0.3*t)*np.cos(np.pi*t)*torch.ones(1,self.nu)
        # # White noise input
        # u = self.noise[int(torch.floor(t*10))].unsqueeze(0)

        # # Step Evolution
        # u = 10.*torch.ones(1,self.nu)
        return u