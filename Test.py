import torch
import torchvision

t = torch.randn(5,3)
print(t)
print("Cuda available: ", torch.cuda.is_available())