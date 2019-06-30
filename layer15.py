
"""
This is layer15 of the DPN implementation for the Biosnake lab
Author: Tim Peisker
input: 21 feature maps of size 512x512 from layer 11 and 21 feature maps of size 512x512 from layer 14
output: 21 feature maps of size 512x512

Brief description:
This final layer combines the output of layer 11 and layer 14 with softmax activation
Probability of assigning label u to pixel at r,c is normalized over all the labels
A visualization of each pixel's final category is also supplied in the end

"""
from __future__ import division
import torch
import numpy as np
import matplotlib.pyplot as plt

# for testing purposes I reduced the mock image size because my computer is too slow otherwise
rows = 20  # total number of rows in one image
cols = 20  # total number of columns in one image
# input should be of the format rows x columns x category channel
data11 = torch.rand(rows,cols,21)  # change this to get the output of layer 11
data14 = torch.rand(rows,cols,21)  # change this to get the output of layer 14

data15 = torch.empty(rows, cols, 21)  # container for the output feature maps
finalOutput = torch.empty(rows, cols)  # this container will store the index of the category with max label prob


# when the input image has more pixels it is important that the for loops below are optimized for parallelization
for r in range(rows):
    for c in range(cols):
        nums = torch.empty(21)
        total = 0
        for u in range(21):
            nums[u] = np.exp(np.log(data11[r,c,u])-data14[r,c,u])  # combine layer 11 and layer 14 results
            total += nums[u]
        for u in range(21):
            data15[r,c,u] = nums[u]/total  # normalization
        finalOutput[r,c] = np.argmax(nums)

# for real images there should be a better way to assign colors to the different categories, but for this mock data a
# simple colorbar scale will suffice
plt.figure()
plt.imshow(finalOutput)
plt.colorbar()
plt.show()
print("end")
