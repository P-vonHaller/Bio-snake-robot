#!/usr/bin/env python
# coding: utf-8

# # Snake Robot - Image Segmentation Task
# ## Documentation
# ### Feussner, Robert ... Last names!!!

# We use 
# - Pycharm mainly for prototyping/ generating code.
# - Jyputer for documentation
# - Pytorch as the main library for computing

# Goal: Reimplement the paper with a ResNet50 instead of a VGG$_{16}$

# First off, we have to get a possible ResNet50 architechture for our task. We can find several models, including the ResNet50 in the _TORCHVISION.MODELS_ package. The sorce code for the ResNet50 can be found here:  
# + https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html#resnet50  

# In[7]:


import torchvision.models as models
# resnet50 = models.resnet50()
# This would initialize the Net randomly, it would have to be trained from scratch

resnet50 = models.resnet50(pretrained=True)
# The pretained Net is already trained on the Imagenet Dataset with over 1,2 Million images 
# and classifies 1000 different classes 


# The on the ImageNet pretrained tensors are downloaded (via the source code from above) from:  
# - https://download.pytorch.org/models/resnet50-19c8e357.pth  

# Now we have the ResNet50 ready for general classification. An example of said classification can be seen below:

# In[1]:


# Juat an example of a picture being classified


# In[ ]:





# In[8]:


print(resnet50)


# In[2]:


import torch
torch.cuda.is_available()


# In[9]:


# Saving a model
torch.save(resnet50, "ResNet50.pt")


# In[10]:


# Loading a model
res = torch.load("ResNet50.pt")
res.eval()


# In[ ]:




