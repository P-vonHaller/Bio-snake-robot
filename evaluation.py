
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
b15_location = parent + "/VOC12_After_b15/"


def main():
    print("Calculating accuracy...")

    print("Done")


if __name__ == "__main__":
    main()
