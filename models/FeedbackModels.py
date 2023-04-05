import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os

from torchdiffeq import odeint_adjoint as odeint

class System_contractive(nn.Module):
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
        std = 0.2           #standard deviation used to draw randomly the initial weights of the model.
        #Initialization of the Free Matrices:
        self.Pstar = nn.Parameter(torch.randn(nx,nx,device=device)*std)
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
        # self.Lambda = torch.zeros(nq,nq)
        self.D11 = torch.zeros(nq,nq,device=device) 
        self.C1 = torch.zeros(nq,nx,device=device)
        self.B1 = torch.zeros(nx,nq,device=device)
        self.P = torch.zeros(nx,nx,device=device)
        self.alpha = alpha
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
        yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_,yi

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

class System_robust_L2_bound(nn.Module):
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
        self.X = nn.Parameter(torch.randn(nx+nq,nx+nq,device=device)*std)    # REMEMBER TO CHANGE IT FOR ROBUST SYSTEMS
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
        yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_,yi

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

class System_robust_passive_input(nn.Module):
    def __init__(self, nx, ny, nu, nq,sigma, epsilon, S, Q, R, ni,device, bias = False, alpha = 0.0):
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
        self.alpha = alpha
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
        yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_,yi

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

class System_robust_passive_output(nn.Module):
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
        yi = F.linear(xi, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22) + By
        return xi_,yi

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

class NODE_REN(nn.Module):
    def __init__(self, nx = 5, ny = 5, nu = 5, nq = 5, sigma = "tanh", epsilon = 1.0e-2, mode = "c", gamma = 1., device = "cpu", bias = False, ni = 1., rho = 1., alpha = 0.0):
        """Base class for Neural Ordinary Differential Equation Recurrent Equilbrium Networks (NODE_RENs).
        
        Args:
            -nx (int): no. internal-states
            -ny (int): no. output
            -nu (int): no. inputs
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'. Defaults to "tanh".
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. Defaults to 1.0e-2.
            -mode (str, optional): Property to ensure. Possible options: 'c'= contractive model, 'rl2'=L2 lipschitz-bounded, 'input_p'=input passive model, 'output_p'=output_passive model.
            -gamma (float, optional): If the model is L2 lipschitz bounded (i.e., mode == 'c'), gamma is the L2 Lipschitz constant. Defaults to 1..
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -ni  (float, optional): If the model is input passive (i.e., mode == 'input_p') , ni is the weight coefficient that characterizes the (input passive) supply rate function.
            -rho (float, optional): If the model is output passive (i.e., mode == 'output_p'), rho is the weight coefficient that characterizes the (output passive) supply rate function.
            -alpha (float, optional): Lower bound of the Contraction rate. If alpha is set to 0, the system continues to be contractive, but with a generic (small) rate. Defaults to 0. 
            """
        super().__init__()
        self.mode = mode.lower()
        if (self.mode == "c"):
            self.RENsys = System_contractive(nx, ny, nu, nq,sigma, epsilon, device=device, bias=bias, alpha= alpha)
        elif(self.mode == "rl2"):
            Q = -np.sqrt(1/gamma)*torch.eye(ny, device=device)
            R = np.sqrt(gamma)*torch.eye(nu,device=device)
            S = torch.zeros(nu,ny,device=device)
            self.RENsys = System_robust_L2_bound(nx, ny, nu, nq,sigma, epsilon,S=S,Q=Q,R=R,gamma=gamma,device=device, bias=bias,alpha=alpha)
        elif(self.mode == "input_p"):
            if (ny != nu):
                raise NameError("u and y have different dimensions, so you cannot have passivity!")
            Q = torch.zeros(ny,device=device)
            R = -2.*ni*torch.eye(nu,device=device)
            S = torch.eye(ny,device=device)
            self.RENsys = System_robust_passive_input(nx, ny, nu, nq,sigma, epsilon,S=S,Q=Q,R=R,ni=ni,device=device,bias=bias,alpha=alpha)
        elif(self.mode == "output_p"):
            if (ny != nu):
                raise NameError("u and y have different dimensions, so you cannot have passivity!")
            Q = -2.*rho*torch.eye(ny,device=device)
            R = torch.zeros(nu,device=device)
            S = torch.eye(ny,device=device)
            self.RENsys = System_robust_passive_output(nx, ny, nu, nq,sigma, epsilon,S=S,Q=Q,R=R,rho=rho,device=device,bias=bias,alpha=alpha)
        else:
            raise NameError("The inserted mode is not valid. Please write 'c', 'rl2','input_p' or 'output_p'. :(")

    def updateParameters(self):
        self.RENsys.updateParameters()

    def forward(self, t, x_REN, y, xbar):
        # error = xbar-y
        error = y #- zeros( ... ). Target (final) velocities= zeros !
        xdot_REN,output_REN = self.RENsys(t, x_REN, error)
        return xdot_REN, output_REN
    
    def output(self, x_REN, y, xbar):
        # error = xbar-x_sys
        error = y #- zeros( ... ). Target (final) velocities= zeros !
        output_REN = self.RENsys.output(x_REN, error)
        return output_REN


class FeedbackSystem(nn.Module):
    def __init__(self, nx=5, nq=5, sigma="tanh", epsilon=3.0e-2, mode = "c", sys = "corridor", n_agents = 2, gamma = 1,ni = 1.0, rho = 1., device="cpu", bias= False,alpha=0.0):
        """Implementation of a negative feedback interconnection with a REN-ODE (as Controller) and a multi-agent systems.  
        
        Args:
            -nx (int): no. internal-states
            -nq (int): no. non-linear states
            -sigma (string): activation function. It is possible to choose: 'tanh','sigmoid','relu','identity'. Defaults to "tanh".
            -epsilon (float): small (positive) scalar used to guarantee the matrices to be positive definitive. Defaults to 1.0e-2.
            -mode (str, optional): Property to ensure. Possible options: 'c'= contractive model, 'rl2'=L2 lipschitz-bounded, 'input_p'=input passive model, 'output_p'=output_passive model.
            -sys (str, optional): Type of system to be controlled. At the time being, only the "corridor" has been implemented. Defaults to "corridor". 
            -n_agents (int, optional): number of agents that compose the system to be controlled. Defaults to 2.
            -gamma (float, optional): If the model is L2 lipschitz bounded (i.e., mode == 'c'), gamma is the L2 Lipschitz constant. Defaults to 1..
            -ni  (float, optional): If the model is input passive (i.e., mode == 'input_p') , ni is the weight coefficient that characterizes the (input passive) supply rate function.
            -rho (float, optional): If the model is output passive (i.e., mode == 'output_p'), rho is the weight coefficient that characterizes the (output passive) supply rate function.
            -device (string): device to be used for the computations using Pytorch (e.g., 'cpu', 'cuda:0',...)
            -bias (bool, optional): choose if the model has non-null biases. Defaults to False.
            -alpha (float, optional): Lower bound of the Contraction rate. If alpha is set to 0, the system continues to be contractive, but with a generic (small) rate. Defaults to 0. 
            """
        super().__init__()
        #Saving informations
        self.n_states_REN = nx               #they are hyper parameters 
        self.n_outputs_REN= n_agents * 2     #number of inputs of sys
        self.n_inputs_REN = n_agents * 2     #number of outputs of sys ( y == x)
        self.n_agents = n_agents
        self.n_nl_states_REN = nq
        self.experiment = sys
        self.device = device
        #internal modules
        self.REN = NODE_REN(nx=self.n_states_REN,ny=self.n_outputs_REN,nu=self.n_inputs_REN,
                          nq=self.n_nl_states_REN,sigma = sigma,epsilon = epsilon,mode =mode, gamma=gamma, ni=ni, rho = rho, device=device, bias=bias,alpha=alpha)               
        self.sys =SystemRobotsPassive(n_agents,linear=True, device = device)

    def updateParameters(self):
        self.REN.updateParameters()


    def forward(self, t, x):
        x_REN, x_sys = torch.split(x,[self.n_states_REN,self.n_agents*4],dim=1)
        #y = self.sys.output(x_sys)   # y==x_sys   !!!!     
        y = F.linear(x_sys,torch.tensor(((0.,0.,0.,0.),(0.,0.,0.,0.),(1.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,0.,0.),(0.,0.,0.,0.),(0.,0.,1.,0.),(0.,0.,0.,1.)), device= self.device).T)
        xdot_REN, u = self.REN(t, x_REN, y, self.xbar)                                         
        xdot_sys = self.sys(t,x_sys,u,self.xbar)                                                              #x_sys = [nt,n_agents*4]     u = [nt, n_agents*2]
        return torch.cat((xdot_REN,xdot_sys),dim=1 )

    def input_evaluation(self,x_tot):
        #it will be called "n_steps" times so your input is just a                                  xtot = [ n_exp , ( nx_ren + 4*n_robots ) ] 
        x_REN, x_sys = torch.split(x_tot,[self.n_states_REN,self.n_agents*4],dim=1)
        #y == x_sys
        y = F.linear(x_sys,torch.tensor(((0.,0.,0.,0.),(0.,0.,0.,0.),(1.,0.,0.,0.),(0.,1.,0.,0.),(0.,0.,0.,0.),(0.,0.,0.,0.),(0.,0.,1.,0.),(0.,0.,0.,1.)), device= self.device).T)
        output_REN = self.REN.output(x_REN, y, self.xbar)                        #u should be a output_REN = [n_exp, 2*n_robots ]
        return output_REN 
    
    def set_reference(self,value):
        self.xbar = value
    
class SystemRobotsPassive(nn.Module):
    def __init__(self,n_agents, device = "cpu",linear=True):
        super().__init__()
        self.n_agents = n_agents
        self.n = 4*self.n_agents
        self.m = 2*self.n_agents
        self.mass = 1.0
        self.b1 = 3.0
        self.k  = .5
        self.device = device
        self.linear = linear
        if linear:
            self.b2 = 0
        else:
            self.b2 = 0.1
        m = self.mass
        self.B = torch.kron(torch.eye(self.n_agents, device= self.device),
                            torch.tensor([[0, 0],
                                          [0, 0],
                                          [1/m, 0],
                                          [0, 1/m]], device= self.device)
                            )
        if linear:
            b1 = self.b1
            m  = self.mass
            k = self.k
            A1 = torch.cat((torch.cat((torch.zeros(2,2, device= self.device),
                                    torch.eye(2, device= self.device)
                                    ), dim=1),
                        torch.cat((torch.diag(torch.tensor([-k/m, -k/m], device= self.device)),
                                    torch.diag(torch.tensor([-b1/m, -b1/m], device= self.device))
                                    ),dim=1),
                        ),dim=0)
            self.A1 = torch.kron(torch.eye(self.n_agents, device= self.device), A1)
        
    def A(self, x):
        # For the time being it is just linear. The non-linear part will be implemented in the future.
        # b2 = self.b2
        if self.linear:
            A1 = self.A1
        else:
            # Here you need to add the nonlinear dependency of A from the state x (not implemented yet!)
            pass
            # b1 = self.b1
            # m  = self.mass
            # k = self.k
            # A1 = torch.cat((torch.cat((torch.zeros(2,2, device= self.device),
            #                             torch.eye(2, device= self.device)
            #                             ), dim=1),
            #                 torch.cat((torch.diag(torch.tensor([-k/m, -k/m], device= self.device)),
            #                             torch.diag(torch.tensor([-b1/m, -b1/m], device= self.device))
            #                             ),dim=1),
            #                 ),dim=0)
            # A1 = torch.kron(torch.eye(self.n_agents, device= self.device), A1)
        #mask = torch.tensor([[0, 0], [1, 1]]).repeat(self.n_agents, 1)
        # A2 = torch.norm(x.view(2 * self.n_agents, 2) * mask, dim=1, keepdim=True)
        # A2 = torch.kron(A2, torch.ones(2,1))
        # A2 = -b2/m * torch.diag(A2.squeeze())        
        A = A1 #+ A2
        return A

    # def forward(self, t, x, u, w):
    def forward(self, t, x, u, xbar):
        # x_ = self.A(x) + w  # here we can add noise not modelled
        x_ = F.linear(x-xbar,self.A(x-xbar)) + F.linear(u,self.B)  # here we can add noise not modelled
        return x_
    
    
    

# class SystemRobots(nn.Module):
#     def __init__(self,n_agents, linear=True):
#         super().__init__()
#         self.n_agents = n_agents
#         self.n = 4*self.n_agents
#         self.m = 2*self.n_agents
#         self.mass = 1.0
#         self.b = 3.0
#         if linear:
#             self.b2 = 0
#         else:
#             self.b2 = 0.1
#         m = self.mass
#         self.B = torch.kron(torch.eye(self.n_agents),
#                             torch.tensor([[0, 0],
#                                           [0, 0],
#                                           [1/m, 0],
#                                           [0, 1/m]])
#                             )
        
#     def A(self, x):
#         # For the time being it is just linear. The non-linear part will be implemented in the future.
#         #b2 = self.b2
#         b1 = self.b
#         m  = self.mass
#         A1 = torch.eye(4*self.n_agents)
#         A2 = torch.cat((torch.cat((torch.zeros(2,2),
#                                     torch.eye(2)
#                                     ), dim=1),
#                         torch.cat((torch.diag(torch.tensor([0, 0])),
#                                     torch.diag(torch.tensor([-b1/m, -b1/m]))
#                                     ),dim=1),
#                         ),dim=0)
#         A2 = torch.kron(torch.eye(self.n_agents), A2)
#         A = A2
#         return A

#     # def forward(self, t, x, u, w):
#     def forward(self, t, x, u):
#         # x_ = self.A(x) + w  # here we can add noise not modelled
#         x_ = F.linear(x,self.A(x)) + F.linear(u,self.B)  # here we can add noise not modelled
#         return x_