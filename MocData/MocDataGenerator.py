#!/usr/bin/env python
# coding: utf-8

# In[53]:


import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
#from datasets import VOCDataSet

import os
import os.path as osp
#import numpy as np
import random
import matplotlib.pyplot as plt
import collections
#import torch
import torchvision
import cv2
from torch.utils import data


# # Change code for different OS (\ or /)

# In[83]:


class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        print(root)
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:	
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        counter = 0 #for testing
        for name in self.img_ids:
            counter += 1
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name) #change / and \ according to os!!!
            print(img_file)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)#change / and \ according to os!!!
            print(img_file)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
            if counter > 10:
                break

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        #print('getting data')
        datafiles = self.files[index]
        #print(datafiles)
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        #print(label)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name


# In[112]:


DATA_DIRECTORY = 'root/VOCdevkit/VOC2012/'
DATA_LIST_PATH = 'train_aug.txt'

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) 

NUM_STEPS = 20000
BATCH_SIZE = 5
INPUT_SIZE = (512,512)


print(DATA_DIRECTORY)


v = VOCDataSet(DATA_DIRECTORY, DATA_LIST_PATH, max_iters=NUM_STEPS*BATCH_SIZE, crop_size=INPUT_SIZE, scale=False, mirror=True, mean=IMG_MEAN)

print(v.__getitem__(0))


# # Somehow only works with num_workers=0, otherwise broken pipe error
# ### At least here...

trainloader = data.DataLoader(VOCDataSet(DATA_DIRECTORY, DATA_LIST_PATH, max_iters=NUM_STEPS*BATCH_SIZE, crop_size=INPUT_SIZE, scale=True, mirror=True, 
                                         mean=IMG_MEAN), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)


for counter, batch in enumerate(trainloader):
        images, labels, _, _ = batch #list of images and labes and shape? and names of each list, partly as tensors
        print(images.shape)
        print(labels)
        if counter >1:
            break





