import torch
import torch.nn as nn


def loss(x,y,t):
    # Modulus of rigidity
    E = 10*(10**9)
    # Poisson's ratio                         
    neu = 0.3
    # Flexural Rigidity of Plate
    D = (E*(h**3))/(12(1-neu**2))

    # Area of the plate
    A = 1
    # Mass of the plate
    mass = 2
    # Mass per unit area
    lambda = mass/A


    X.requires_grad = True
    Y.requires_grad = True
    T.requires_grad = True

    x_zeroes = torch.zeros_like(x)
    y_zeroes = torch.zeros_like(y)
    T_zeroes = torch.zeros_like(y)
    X_max = 1*torch.ones_like(x)
    Y_max = 1*torch.ones_like(y)

    w = NeuralNetworks(torch.hstack((X, Y, T)))   # w is the displacement

    dw_dx = torch.autograd.grad(w, X, grad_outputs = torch.ones_like(w), create_graph = True)[0]
    d2w_dx2 = torch.autograd.grad(dw_dx, X, grad_outputs = torch.ones_like(dw_dx), create_graph = True)[0]
    d3w_dx3 = torch.autograd.grad(d2w_dx2w, X, grad_outputs = torch.ones_like(d2w_dx2w), create_graph = True)[0]
    d4w_dx4 = torch.autograd.grad(d3w_dx3, X, grad_outputs = torch.ones_like(d3w_dx3w), create_graph = True)[0]
    
    dw_dy = torch.autograd.grad(w, Y, grad_outputs = torch.ones_like(w), create_graph = True)[0]
    d2w_dy2 = torch.autograd.grad(dw_dy, Y, grad_outputs = torch.ones_like(dw_dy), create_graph = True)[0]
    d3w_dy3 = torch.autograd.grad(d2w_dy2w, Y, grad_outputs = torch.ones_like(d2w_dy2w), create_graph = True)[0]
    d4w_dy4 = torch.autograd.grad(d3w_dy3, Y, grad_outputs = torch.ones_like(d3w_dy3), create_graph = True)[0]
    
    d3w_dx2dy = torch.autograd.grad(d2w_dx2, Y, grad_outputs = torch.ones_like(d2w_dx2), create_graph = True)[0]
    d4w_dx2dy2 = torch.autograd.grad(d3w_dx2dy, Y, grad_outputs = torch.ones_like(d3w_dx2dy), create_graph = True)[0]    
   
    dw_dt = torch.autograd.grad(w, T, grad_outputs = torch.ones_like(w), create_graph = True)[0] 
    d2w_dt2 = torch.autograd.grad(dw_dt, T, grad_outputs = torch.ones_like(dw_dt), create_graph = True)[0]
    
    f = torch.mean((D*(d4w_dx4+d4w_dx2dy2+d4w_dy4)+rho*d2w_dt2)**2)
    g = NeuralNetworks(torch.hstack((x_zeroes, Y, T)))
    h = NeuralNetworks(torch.hstack((X, y_zeroes, T)))
    l = NeuralNetworks(torch.hstack((X_max, Y, T)))
    m = NeuralNetworks(torch.hstack((X, Y_max, T)))
    n = NeuralNetworks(torch.hstack((X,Y_max, T_zeroes))

    mse = torch.mean((g - 0)**2) + torch.mean((h - 0)**2) + torch.mean((l - 0)**2) + torch.mean((m - 0)**2) + torch.mean((n-0)**2)
    loss = f + mse

    return loss


optimizer = torch.optim.LBFGS(N.parameters())
x = torch.linspace(0,1,100)
y = torch.linspace(0,1,100)
t = torch.linspace(0,1,100)
X, Y, T = torch.meshgrid(x, y, t)

def closure():
    optimizer.zero_grad()
    l = loss(x)
    l.backward()
    return l

epochs = 100000
for i in range(epochs):
    optimizer.step(closure)







