# jimmy shen
# April 1, 2018
import torch
from torch.autograd import Variable
# encoding=utf8
import sys
import time
start = time.time()
print(sys.stdout.encoding)
N, D = 20000, 20000

x = Variable(torch.randn(N,D).cuda(), requires_grad = True)
y = Variable(torch.randn(N,D).cuda(), requires_grad = True)
z = Variable(torch.randn(N,D).cuda(), requires_grad = True)

a = x*y
print(a.size())
print(a)
b = a + z
c = torch.sum(b)

#c.backward()
#print(a)
end = time.time()
print("time used for execution this code is: ",1000*(end-start), "ms")
#print(x.grad.data)
#print(y.grad.data)
#print(z.grad.data)

