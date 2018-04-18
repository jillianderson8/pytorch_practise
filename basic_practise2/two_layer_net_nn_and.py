# -*- coding: utf-8 -*-
"""
PyTorch: Tensors
----------------

A fully-connected ReLU network with one hidden layer and no biases, trained to
predict y from x by minimizing squared Euclidean distance.

This implementation uses PyTorch tensors to manually compute the forward pass,
loss, and backward pass.

A PyTorch Tensor is basically the same as a numpy array: it does not know
anything about deep learning or computational graphs or gradients, and is just
a generic n-dimensional array to be used for arbitrary numeric computation.

The biggest difference between a numpy array and a PyTorch Tensor is that
a PyTorch Tensor can run on either CPU or GPU. To run operations on the GPU,
just cast the Tensor to a cuda datatype.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimensionN, D_in, H, H2,D_out = 6400, 2, 16,8, 1
N, D_in, H, D_out = 12800, 2, 20, 1
# Create random input and output data
#x = torch.randn(N, D_in).type(dtype)
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = nn.MSELoss(size_average=False)

learning_rate = 1e-5
for mini_patch in range(10):
    learning_rate/=(mini_patch+1)
    x_ = torch.randn(N, D_in).type(dtype)
    y = torch.zeros(N, D_out).type(dtype)
    x =torch.zeros(N, D_in).type(dtype)
    x[x_ > 0.5] = 1.0
    
    for i in range(N):
        y[i,0] = x[i,0]*x[i,1]
        # print(x[i,0],x[i,1], y[i,0])
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
x_new_ = torch.randn(N, D_in).type(dtype)
x_new =torch.zeros(N, D_in).type(dtype)
x_new[x_new_ > 0.5] = 1.0
x_new=Variable(x_new,requires_grad=False)
y_pred = model(x_new)

for i in range(100):
    print(float(x_new[i,0])," and ",float(x_new[i,1])," is ", float(y_pred[i,0]))
