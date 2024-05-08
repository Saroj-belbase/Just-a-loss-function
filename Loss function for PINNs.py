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


    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True

    x_zeroes = torch.zeros_like(x)
    y_zeroes = torch.zeros_like(y)
    X_max = 1*torch.ones_like(x)
    Y_max = 1*torch.ones_like(y)

    w = NeuralNetworks(torch.hstack((x, y, t)))   # w is the displacement

    dw_dx = torch.autograd.grad(w, x, grad_outputs = torch.ones_like(w), create_graph = True)[0]
    d2w_dx2 = torch.autograd.grad(dw_dx, x, grad_outputs = torch.ones_like(dw_dx), create_graph = True)[0]
    d3w_dx3 = torch.autograd.grad(d2w_dx2w, x, grad_outputs = torch.ones_like(d2w_dx2w), create_graph = True)[0]
    d4w_dx4 = torch.autograd.grad(d3w_dx3, x, grad_outputs = torch.ones_like(d3w_dx3w), create_graph = True)[0]
    
    dw_dy = torch.autograd.grad(w, y, grad_outputs = torch.ones_like(w), create_graph = True)[0]
    d2w_dy2 = torch.autograd.grad(dw_dy, x, grad_outputs = torch.ones_like(dw_dy), create_graph = True)[0]
    d3w_dy3 = torch.autograd.grad(d2w_dy2w, x, grad_outputs = torch.ones_like(d2w_dy2w), create_graph = True)[0]
    d4w_dy4 = torch.autograd.grad(d3w_dy3, x, grad_outputs = torch.ones_like(d3w_dy3), create_graph = True)[0]
    
    d3w_dx2dy = torch.autograd.grad(d2w_dx2, x, grad_outputs = torch.ones_like(d2w_dx2), create_graph = True)[0]
    d4w_dx2dy2 = torch.autograd.grad(d3w_dx2dy, x, grad_outputs = torch.ones_like(d3w_dx2dy), create_graph = True)[0]    
   
    dw_dt = torch.autograd.grad(w, t, grad_outputs = torch.ones_like(w), create_graph = True)[0] 
    d2w_dt2 = torch.autograd.grad(dw_dt, t, grad_outputs = torch.ones_like(dw_dt), create_graph = True)[0]
    
    f = torch.mean((D*(d4w_dx4+d4w_dx2dy2+d4w_dy4)+rho*d2w_dt2)**2)
    g = NeuralNetworks(torch.hstack((x_zeroes, y, t)))
    h = NeuralNetworks(torch.hstack((x, y_zeroes, t)))
    l = NeuralNetworks(torch.hstack((X_max, y, t)))
    m = NeuralNetworks(torch.hstack((x, Y_max, t)))

    mse = torch.mean((g - 0)**2) + torch.mean((h - 0)**2) + torch.mean((l - 0)**2) + torch.mean((m - 0)**2)
    loss = f + mse

    return loss


optimizer = torch.optim.LBFGS(N.parameters())
x = torch.linspace(0,1,100)
y = torch.linspace(0,1,100)
t = torch.linspace(0,1,100)

def closure():
    optimizer.zero_grad()
    l = loss(x)
    l.backward()
    return l

epochs = 100000
for i in range(epochs):
    optimizer.step(closure)







