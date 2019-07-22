
"""
This is a small script to evaluate the DPN implementation for the Biosnake lab
Author: Tim Peisker
input: load VOC12_After_b15 data test data and corresponding ground truth
output: average accuracy over test data

Brief description:
This script loads the test data after it has been subjected to our DPN implementation. We compare the labels assigned
to each pixel with the labels from ground truth to report and accuracy on the test data

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
b15_location = parent + "/VOC12_After_b15/"
num_train = 1400  # number of training image batches


def main():
    print("Calculating accuracy...")

    acc_sum = 0
    for i in range(num_train):  # loop over all the batches
        # load the test data segmentation from our network
        dpn = torch.load(b15_location + "/test" + str(i) + ".pth").to(device)
        gt = torch.load(b11_location + "/test-labels" + str(i) + ".pth").to(device)  # load ground truth
        acc_sum += evaluate(dpn, gt)  # compute accuracy of this batch and sum over all the batches
    acc = acc_sum/num_train  # divide the sum of accuracies by the number of batches to get average accuracy
    print("The accuracy on the test data is: " + acc)
    print("Done")


def evaluate():
    # TODO: insert evaluation function here
    return 0 # return the accuracy of this batch


if __name__ == "__main__":
    main()
