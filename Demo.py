import random
import matplotlib.pyplot as plt
import collections
import torchvision
from torch.utils import data

from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import argparse #don't need it, actually...

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp

sys.path.append('Pytorch-Deeplab') # needed for the next 3 lines

from deeplab.model import Res_Deeplab
from deeplab.loss import CrossEntropy2d
from deeplab.datasets import VOCDataSet
import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()




INPUT_SIZE = '321,321'
PATH = '/root/Bio-snake-robot/1.jpg'

h, w = map(int, INPUT_SIZE.split(','))
input_size = (h, w)
mean=(128, 128, 128)

image = cv2.imread(PATH, cv2.IMREAD_COLOR)
size = image.shape
np.array(size)
image = np.asarray(image, np.float32)
image -= mean
        
img_h, img_w, _ = image.shape
pad_h = max(h - img_h, 0)
pad_w = max(w - img_w, 0)
if pad_h > 0 or pad_w > 0:
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
image = image.transpose((2, 0, 1))
image = torch.from_numpy(image)

image = image.unsqueeze(0)

gpu0 = 0
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu0)
cudnn.enabled = True
model = Res_Deeplab(num_classes=21)
pathToTrainedModel = '/root/Bio-snake-robot/Pytorch-Deeplab/FirstTrain10000StepsDefaultParametersBatch6/VOC12_scenes_10000.pth'
saved_state_dict = torch.load(pathToTrainedModel)
model.load_state_dict(saved_state_dict)
model.eval()
model.cuda()

image = Variable(image).cuda() #gets and saves a gpu output, for cpu see evaluate.py

interp = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

pred = interp(model(image))
output = pred.cpu().data[0].numpy()

output = output[:,:size[0],:size[1]]
output = output.transpose(1,2,0)
output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

fig, ax = plt.subplots()

classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5), 
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5), 
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0), 
                    (0.5,0.75,0),(0,0.25,0.5)]
cmap = colors.ListedColormap(colormap)
bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
norm = colors.BoundaryNorm(bounds, cmap.N)

ax.set_title('Prediction')
ax.imshow(pred, cmap=cmap, norm=norm)

fig.savefig('/root/ForTesting/Hurray.png')






















