
"""
This is layer15 of the DPN implementation for the Biosnake lab
Author: Tim Peisker
input: 21 feature maps of size 321x321 from layer 11 and 21 feature maps of size 321x321 from layer 14
output: 21 feature maps of size 321x321

Brief description:
This final layer combines the output of layer 11 and layer 14 with softmax activation
Probability of assigning label u to pixel at r,c is normalized over all the labels

"""
from __future__ import division
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dirpath = os.getcwd()
parent = os.path.dirname(dirpath)
b11_location = parent + "/VOC12_After_Deeplab/TrainBatch3TensorsGPU"
b14_location = parent + "/VOC12_After_b14/TrainBatch3TensorsGPU"
output_location = parent + "/VOC12_After_b15"



def main():

    # display paths to the console

    print("The current path is: " + dirpath)
    print("This is the parent directory: " + parent)
    print("This is the location of the b11 data: " + b11_location)
    print("This is the location of the b14 data: " + b14_location)
    print("This is the location where the b15 output will be stored: " + output_location)

    # Note: the output of an image after b11 is assumed to be saved with the same name as the output after b14 for that
    #  image within their respective folders.

    for i in range(3525):  # loop over all the batches

        # both data sources are 3x21x321x321 (batch size x categories x downsampled image dimensions)
        b11 = torch.load(b11_location + "/predictions" + str(i) + ".pth")
        #print(b11.size())
        b14 = torch.load(b14_location + "/predictions" + str(i) + ".pth")
        #print(b14.size())
        b15 = torch.zeros([3, 21, 321, 321])  # tensor to store output of this batch

        for j in range(3):  # loop over the individual 3D tensors within a batch
            den = torch.zeros([321, 321])  # variable to sum over the categories to get the denominator of Eqn. 15
            for k in range(21):  # loop over all the categories within one 3D tensor
                fmap11 = b11[j, k].to(device)  # kth category in the jth feature map of this batch
                fmap14 = b14[j, k].to(device)

                num = torch.exp(torch.log(fmap11)-fmap14)  # compute the numerator of Eqn. 15
                b15[j,k] = num  # store the numerator so that we can later divide by the denominator
                den = den.to(device) + num.to(device)  # summing over all the categories to get den
            for k in range(21):
                b15[j,k] = b15[j,k].to(device)/den.to(device)  # divide every category's fmap by the denominator

        torch.save(b15, output_location + "/predictions" + str(i) + ".pth")
        #if i == 1: break
        print(i)




def test():

    # for testing purposes I reduced the mock image size because my computer is too slow otherwise
    rows = 20  # total number of rows in one image
    cols = 20  # total number of columns in one image
    # input should be of the format rows x columns x category channel
    data11 = torch.rand(rows, cols, 21)  # change this to get the output of layer 11
    data14 = torch.rand(rows, cols, 21)  # change this to get the output of layer 14

    data15 = torch.empty(rows, cols, 21)  # container for the output feature maps
    finalOutput = torch.empty(rows, cols)  # this container will store the index of the category with max label prob

    # when the input image has more pixels it is important that the for loops below are optimized for parallelization
    for r in range(rows):
        for c in range(cols):
            nums = torch.empty(21)
            total = 0
            for u in range(21):
                nums[u] = np.exp(np.log(data11[r, c, u]) - data14[r, c, u])  # combine layer 11 and layer 14 results
                total += nums[u]
            for u in range(21):
                data15[r, c, u] = nums[u] / total  # normalization
            finalOutput[r, c] = np.argmax(nums)

    # for real images there should be a better way to assign colors to the different categories, but for this mock data a
    # simple colorbar scale will suffice
    plt.figure()
    plt.imshow(finalOutput)
    plt.colorbar()
    plt.show()
    print("end")

if __name__ == "__main__":
    main()
    #write_sample_data()