import torch
import torchvision

t = torch.randn(5,3)
print(t)
print("Cuda available: ", torch.cuda.is_available())
f = open('Test.txt', 'w+')
f.write(t)
f.write("Cuda available: ", torch.cuda.is_available())
f.close()