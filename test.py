import os

import scipy.io as scio

# l=[]
# with open('G:/LV-MHP-v2/list/test_all.txt', 'r') as f:
#     while True:
#         a=f.readline()
#         a=a.strip()
#         if not a:
#           break
#         l.append(a)
#     print(l)

import scipy

# a=scio.loadmat('G:/LV-MHP-v2/val/pose_annos/25500.mat')
#
# for x in a.keys():
#     print(x)
#     print(a[x])

# print(  [ "%s_%05d"%('55',x)for x in range(1,2+1) ]  )
# from PIL import Image
# import numpy as np
# a=Image.open('G:/LV-MHP-v2/train/parsing_annos/4_02_02.png')
# b= np.array(a)
# print( np.max(b[:,:,0]))

import torch
import torch.nn as nn

class ABC(nn.Module):
    def __init__(self):
        super(ABC, self).__init__()
        self.n1=nn.Linear(3,3)
    def forward(self, x):
        n=self.n1(x)
        return n
with torch.no_grad():
    b=ABC()
class CCC(nn.Module):
    def __init__(self):
        super(CCC, self).__init__()

        self.n1=nn.Linear(3,1)
    def forward(self, x):
        n=b(x)
        n=self.n1(n)
        return n
class CCD(nn.Module):
    def __init__(self):
        super(CCD, self).__init__()
        self.z=b
        self.n1=nn.Linear(3,1)
    def forward(self, x):
        n=self.z(x)
        n=self.n1(n)
        return n
c=CCC()
d=CCD()
import torch.optim as optim
optimizer1 = optim.SGD(c.parameters(), lr=0.01, momentum=0.9)
optimizer2 = optim.SGD(d.parameters(), lr=0.01, momentum=0.9)
optimizer3 = optim.SGD(b.parameters(), lr=0.01, momentum=0.9)
AAA=torch.rand((2,3))
e=c(AAA)
f=d(AAA)
print(e)
h=torch.rand((2,1))
loss=nn.MSELoss()

losses1=loss(e,h)

losses2=loss(f,h)
losses1.backward()
for x in c.parameters():
    print(x)
print('----------')
for x in d.parameters():
    print(x)
print('----------')