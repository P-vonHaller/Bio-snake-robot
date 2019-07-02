"""
A simple data loader for images
"""

import torch
import torchvision
from torch.utils import data
from matplotlib import pyplot as plt
import numpy as np
# credit to: https://stackoverflow.com/questions/50052295/how-do-you-load-images-into-pytorch-dataloader


def load_dataset():
    data_path = 'data'
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

if __name__== '__main__':

    images = load_dataset()
    for i, data in enumerate (images, 1):
        print np.shape(data[0][0].numpy())
        plt.imshow(data[0][0].numpy())
        plt.show()