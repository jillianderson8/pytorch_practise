# The cifar experiment is done based on the tutorial provided by 
http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py


the first version is exactly the same one as shown in the tutorial

the gpu version is changed from without padding to padding to padding+deeper network

the resnet18 is based on the resnet 18 with and without pretrain also frozen the conv parameters and unfrozen the parameters of the conv layer. detail is given as below:

| File Name                                 | pretrain?    |epoch|
| ------------------------------------------|------------- |-----|
| cifar_resnet18_no_pretrain.py             | no           |10   |
| cifar_resnet18_pretrain.py                | yes          |10   |
| cifar_resnet18_pretrain_30_epoch.py       | yes          |30   |
|cifar_resnet18_pretrain_unfrozen_30epoch.py|  yes         |30   |
