
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
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os



def main():

    # get paths and display them to the console
    dirpath = os.getcwd()
    parent = os.path.dirname(dirpath)
    b11_location = parent + "/VOC12_After_Deeplab/TrainBatch3TensorsGPU"
    b14_location = parent + "/data"  # parent + "/VOC12_After_b14/TrainBatch3TensorsGPU"
    output_location = parent + "/VOC12_After_b15"
    print("The current path is: " + dirpath)
    print("This is the parent directory: " + parent)
    print("This is the location of the b11 data: " + b11_location)
    print("This is the location of the b14 data: " + b14_location)
    print("This is the location where my ouput will be stored: " + output_location)


    # the output of an image after b11 is assumed to be saved with the same name as the output after b14 for that image
    # within their respective folders.
    counter = 0
    for b14_file in b14_location:
        print "b14: " + b14_file
        b11_file = os.path.abspath(b14_file)
        print type(b11_file)
        b11 = load_b11(b11_location)
        print "b11 loader returned type: "+ type(b11)
        if counter == 10: break
        counter +=1

def load_b11(dest , ext = ".pth", bat_size = 2):

    data_path = dest

    train_dataset = torchvision.datasets.DatasetFolder(
        root=data_path,
        loader="/root/VOC12_After_Deeplab/TrainBatch3TensorsGPU/predictions999.pth",
        extensions=ext,
        transform=torchvision.transforms.ToTensor()
    )
    return train_dataset

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