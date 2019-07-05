#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
Module = nn.Module
import collections
from itertools import repeat

unfold = F.unfold


# In[2]:


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


# In[3]:


def conv2d_local(input, weight, bias=None, padding=0, stride=1, dilation=1):
    if input.dim() != 4:
        raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(input.dim()))
    if weight.dim() != 6:
        # outH x outW x outC x inC x kH x kW
        raise NotImplementedError("Input Error: Only 6D weight Tensors supported (got {}D)".format(weight.dim()))
 
    outH, outW, outC, inC, kH, kW = weight.size()
    kernel_size = (kH, kW)
 
    # N x [inC * kH * kW] x [outH * outW]
    cols = unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride)
    cols = cols.view(cols.size(0), cols.size(1), cols.size(2), 1).permute(0, 2, 3, 1)
 
    out = torch.matmul(cols, weight.view(outH * outW, outC, inC * kH * kW).permute(0, 2, 1))
    out = out.view(cols.size(0), outH, outW, outC).permute(0, 3, 1, 2)
 
    if bias is not None:
        out = out + bias.expand_as(out)
    return out


# In[4]:


class Conv2dLocal(Module):
 
    def __init__(self, in_height, in_width, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2dLocal, self).__init__()
 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
 
        self.in_height = in_height
        self.in_width = in_width
        self.out_height = int(math.floor(
            (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        self.out_width = int(math.floor(
            (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        self.weight = Parameter(torch.Tensor(
            self.out_height, self.out_width,
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_channels, self.out_height, self.out_width))
        else:
            self.register_parameter('bias', None)
 
        self.reset_parameters()
 
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
 
    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
 
    def forward(self, input):
        return conv2d_local(
            input, self.weight, self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation)


# In[5]:


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        #in_height, in_width, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1
        self.conv1 = Conv2dLocal(5, 5, 21, 21, 3, 1, 1, 0, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x

net = Net()
print(net)
print(net.conv1.weight.data.size())


# In[6]:

def loadImages()
    test = (torch.randint(0, 10, (3, 5, 5))) # color, height, width
    print(test)
    #print(test[1][0][0])
    #print(len(test))


# In[18]:


def calcDistance(color1, color2, color3, compColor1, compColor2, compColor3, imageX, imageY, x, y, middle):
    colorWeight = 0.5
    spatialWeight = 0.5
    distance = 0
    distance += colorWeight * math.sqrt((color1-compColor1)**2 + (color2-compColor2)**2 + (color3-compColor3)**2)
    distance += spatialWeight * math.sqrt((imageX-(imageX-middle+x))**2 + (imageY-(imageY-middle+y))**2)
    return distance


# In[19]:


def initializeFilters(image):
    imageHeight = len(image[0])
    imageWidth = len(image[0][0])
    filterHeight = 3
    filterWidth = 3
    filters = torch.zeros(imageHeight, imageWidth, filterHeight, filterWidth) #imageY, imageX, y, x
    middle = math.floor(filterHeight/2)
    for imageY in range(imageHeight):
        for imageX in range(imageWidth):
            for y in range(filterHeight):
                for x in range(filterWidth):
                    if not ((imageX + (x-middle)) < 0 or (imageX + x) > imageWidth or (imageY + (y-middle)) < 0 or (imageY + y) > imageHeight):
                        filters[imageY][imageX][y][x] = calcDistance(image[0][imageY][imageX], 
                                                                      image[1][imageY][imageX], 
                                                                      image[2][imageY][imageX], 
                                                                      image[0][imageY - middle + y][imageX - middle + x], 
                                                                      image[1][imageY - middle + y][imageX - middle + x], 
                                                                      image[2][imageY - middle + y][imageX - middle + x],
                                                                      imageX, imageY, x, y, middle)
                    else:
                        filters[imageY][imageX][y][x] = 0
    return filters


# In[20]:


filters = initializeFilters(test)
print(filters.size())
#print(filters.view(25, 3, 3).size())
print(filters)
filters.unsqueeze_(-3)
filters.unsqueeze_(-3)
filters = filters.expand(5, 5, 21, 21, 3, 3)
print(filters.size())
#print(filters)


# In[10]:


params = list(net.conv1.parameters())
print(params[0].size())  # conv1's .weight
params = filters
print(params)


# In[21]:


input = torch.ones(21, 21, 5, 5)
print(input[0][0][0][0])
out = net(input)
#print(input)
print(out.size())
#print(out[0][0][0][0])
print(out)



def main():
    #loadimages
    #loadoutput11
    #createfilters
    #applynetwork
    #saveoutput12
    print("test")

if __name__ == "__main__":
   main()