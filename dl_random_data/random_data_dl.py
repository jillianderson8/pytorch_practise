# -*- coding: utf-8 -*-
#based on the tutorial 
# http://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#sphx-glr-beginner-blitz-data-parallel-tutorial-py


import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

input_size = 5
output_size = 2

batch_size = 30
data_size = 100

# generate dummy dataset


class RandomDataset(Dataset):
    
    def __init__(self, size, length):
        self.len = length
        self.data = th.randn(length, size)
        print("the random data set size:",self.data.size())
    def __getitem__(self,index):
        return self.data[index]
    
    def __len__(self):
        return self.len
dataset = RandomDataset(input_size, data_size)
print (dataset.data,type(dataset.data))

rand_loader = DataLoader(dataset = RandomDataset(input_size, data_size), batch_size = batch_size, shuffle = True)

print("rand_loader", rand_loader, len(rand_loader),type(rand_loader))
class Model(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print(" In Model: input size", input.size(), "output size:", output.size())
        return output


model = Model(input_size, output_size)
if th.cuda.is_available():
    print("cuda is availabe")
    model.cuda()

for data in rand_loader:
    if th.cuda.is_available():
        input_var = Variable(data.cuda())
    else:
        input_var = Variable(data)
    output = model(input_var)
    print("Outside: input size", input_var.size(),"output_size:", output.size())
