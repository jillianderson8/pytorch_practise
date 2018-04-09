# -*- coding: utf-8 -*-
"""
PyTorch: Tensors
using three layers to get ride of the xor problem.
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
N, D_in, H1, H2,H3, D_out = 12800, 2, 100, 40, 20, 1
# Create random input and output data
#x = torch.randn(N, D_in).type(dtype)


# Randomly initialize weights
w1 = torch.randn(D_in, H1).type(dtype)
w2 = torch.randn(H1, H2).type(dtype)
w3 = torch.randn(H2, H3).type(dtype)
w4 = torch.randn(H3, D_out).type(dtype)
learning_rate = 5e-8
for mini_patch in range(2000):
    print( "mini patch:", mini_patch)
    learning_rate/=(mini_patch+1)
    x_ = torch.randn(N, D_in).type(dtype)
    y = torch.zeros(N, D_out).type(dtype)
    x =torch.zeros(N, D_in).type(dtype)
    x[x_ > 0.5] = 1.0
    for i in range(N):
        # xor is done here. 0,0 ==> 1 ,
        #                   1,1 ==> 1
        if sum(x[i])<0.5 or sum(x[i])>1.5:
            y[i,0]=1.0
        #y[i,0] = x[i,0]*x[i,1]
        #print(x[i,0],x[i,1], y[i,0])
    for t in range(6000):
    # Forward pass: compute predicted y
        h1 = x.mm(w1)
        h1_relu = h1.clamp(min=0)
        h2 = h1_relu.mm(w2)
        h2_relu = h2.clamp(min=0)
        h3 = h2_relu.mm(w3)
        h3_relu = h3.clamp(min=0)
        y_pred = h3_relu.mm(w4)

    # Compute and print loss
        loss = (y_pred - y).pow(2).sum()
        if t%1000==0:
             print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w4 = h3_relu.t().mm(grad_y_pred)
        grad_h3_relu = grad_y_pred.mm(w4.t())
        grad_h3 = grad_h3_relu.clone()
        grad_h3[h3 < 0] = 0
        grad_w3 = h2_relu.t().mm(grad_h3)
        
        grad_h2_relu = grad_h3.mm(w3.t())
        grad_h2 = grad_h2_relu.clone()
        grad_h2[h2 < 0] = 0
        grad_w2 = h1_relu.t().mm(grad_h2)
        grad_h1_relu = grad_h2.mm(w2.t())
        grad_h1 = grad_h1_relu.clone()
        grad_h1[h1 < 0] = 0
        grad_w1 = x.t().mm(grad_h1)

    # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2 
        w3 -= learning_rate * grad_w3
        w4 -= learning_rate * grad_w4 

x_new_ = torch.randn(N, D_in).type(dtype)
x_new =torch.zeros(N, D_in).type(dtype)
x_new[x_new_ > 0.5] = 1.0




h1 = x_new.mm(w1)
h1_relu = h1.clamp(min=0)
h2 = h1_relu.mm(w2)
h2_relu = h2.clamp(min=0)
h3 = h2_relu.mm(w3)
h3_relu = h3.clamp(min=0)
y_pred = h3_relu.mm(w4)


for i in range(100):
    print(float(x_new[i,0])," and ",float(x_new[i,1])," is ", float(y_pred[i,0]))
