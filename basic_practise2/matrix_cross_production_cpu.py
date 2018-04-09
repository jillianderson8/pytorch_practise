import torch


#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
N=40

# Create random input and output data
A = torch.randn(N, N).type(dtype)
B = torch.randn(N, N).type(dtype)

C = A*B
print(A)
print(B)
print(C)
