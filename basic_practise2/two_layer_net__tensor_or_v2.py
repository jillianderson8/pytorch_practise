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


#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimensionN, D_in, H, H2,D_out = 6400, 2, 16,8, 1
N, D_in, H, D_out = 12800, 2, 20, 1
# Create random input and output data
#x = torch.randn(N, D_in).type(dtype)


# Randomly initialize weights
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)
learning_rate = 1e-6
for mini_patch in range(10):
    learning_rate/=(mini_patch+1)
    x_ = torch.randn(N, D_in).type(dtype)
    y = torch.zeros(N, D_out).type(dtype)
    x =torch.zeros(N, D_in).type(dtype)
    x[x_ > 0.5] = 1.0
    for i in range(N):
        y[i,0] = x[i,0]+x[i,1]
        if y[i,0]>1.5:
            y[i,0] = 1.0
        # print(x[i,0],x[i,1], y[i,0])
    for t in range(10000):
    # Forward pass: compute predicted y
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

    # Compute and print loss
        loss = (y_pred - y).pow(2).sum()
        if t%1000==0:
             print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

x_new_ = torch.randn(N, D_in).type(dtype)
x_new =torch.zeros(N, D_in).type(dtype)
x_new[x_new_ > 0.5] = 1.0


h = x_new.mm(w1)
h_relu = h.clamp(min=0)
y_pred = h_relu.mm(w2)

for i in range(100):
    print(float(x_new[i,0])," and ",float(x_new[i,1])," is ", float(y_pred[i,0]))
