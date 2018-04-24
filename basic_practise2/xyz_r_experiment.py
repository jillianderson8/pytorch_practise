# -*- coding: utf-8 -*-
"""
input: x, y, z
output: x**2+y**2+z**2
verify whether the FC can learn this mapping.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimensionN, D_in, H, H2,D_out = 6400, 2, 16,8, 1
N, D_in, H, D_out = 12800, 3, 40, 1
# Create random input and output data
#x = torch.randn(N, D_in).type(dtype)
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, int(H/2)),
    torch.nn.ReLU(),
    torch.nn.Linear(int(H/2), int(H/4)),
    torch.nn.ReLU(),
    torch.nn.Linear(int(H/4), D_out))

loss_fn = nn.MSELoss(size_average=False)

learning_rate = 1e-5
for mini_patch in range(100):
    learning_rate/=(mini_patch+1)
    x = torch.randn(N, D_in).type(dtype)
    y = torch.zeros(N, D_out).type(dtype)
   # x =torch.zeros(N, D_in).type(dtype)
   # x[x_ > 0.5] = 1.0
    for i in range(N):
       y[i,0] = x[i,0]*2+x[i,1]*2+x[i,2]*2
    
    x, y= Variable(x,requires_grad=False),Variable(y,requires_grad=False)
    for t in range(100):
        y_pred = model(x)
    
    # Compute and print loss. We pass Variables containing the predicted and true
    # values of y, and the loss function returns a Variable containing the
    # loss.
        loss = loss_fn(y_pred, y)
        print(t, loss.data[0])

    # Zero the gradients before running the backward pass.
        model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Variables with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
        loss.backward()

    # Update the weights using gradient descent. Each parameter is a Variable, so
    # we can access its data and gradients like we did before.
        for param in model.parameters():
            param.data -= learning_rate * param.grad.data
x_new = torch.randn(N, D_in).type(dtype)
x_new=Variable(x_new,requires_grad=False)
y_pred = model(x_new)

for i in range(100):
    print(float(x_new[i,0]),float(x_new[i,1]),float(x_new[i,2])," is predicted as ", float(y_pred[i,0]),"gt=",x_new[i,0]**2+x_new[i,1]**2+x_new[i,2]**2)
