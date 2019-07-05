"""
A simple data loader for images
"""

import torch
import torchvision
from torch.utils import data
from matplotlib import pyplot as plt
import numpy as np
# credit to: https://stackoverflow.com/questions/50052295/how-do-you-load-images-into-pytorch-dataloader


def load_images():
    data_path = '/Users/peiskert/Documents/TUM/2019_SS/Lab-Biosnake/data'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )
    return train_loader


# input: destination folder (where is your dataset) and extension (".pt", ".png", etc.), batch size
# output: returns a DataLoader object with which you can load your dataset in batches
def load_dataset(dest = "/home/snakbot/SegmentationTask/VOCdevkit/VOC2012/JPEGImages/" , ext = ".jpeg", bat_size = 50):

    data_path = dest

    train_dataset = torchvision.datasets.DatasetFolder(
        root=data_path,
        loader="/root/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg",
        extensions=ext,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=bat_size,
        num_workers=0,
        shuffle=False
    )
    return train_loader

if __name__== '__main__':

    data = load_dataset()
    for i, item in enumerate(data, 1):
        print np.shape(data[0][0].numpy())

    """
    images = load_images()
    for i, data in enumerate (images, 1):
        print np.shape(data[0][0].numpy())
        plt.imshow(data[0][0][0].numpy())
        plt.show()
    """