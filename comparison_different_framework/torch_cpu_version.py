# jimmy shen
# April 1, 2018
import torch
from torch.autograd import Variable
# encoding=utf8
import sys
print(sys.stdout.encoding)
N, D = 30, 40

x = Variable(torch.randn(N,D), requires_grad = True)
y = Variable(torch.randn(N,D), requires_grad = True)
z = Variable(torch.randn(N,D), requires_grad = True)

a = x*y
print(a.size())
print(a)
b = a + z
c = torch.sum(b)

c.backward()
print(a)
#print(x.grad.data)
#print(y.grad.data)
#print(z.grad.data)

