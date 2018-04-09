import torch as th
from torch.autograd import Variable

x = Variable(th.FloatTensor([1,2,3]), requires_grad = True) 
y = x +2
z = y*y
output = z.mean()
#output.backward()
z.backward(th.FloatTensor([1.0, 1.0, 1.0]))
print(x.grad)
