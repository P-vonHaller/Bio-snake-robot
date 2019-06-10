import torch
import torchvision

t = torch.randn(5,3)
print(t)
print("Cuda available: ", torch.cuda.is_available())
f = open('Test.txt', 'w+')
for i in range(10):
    f.write("This is line %d\r\n" % (i+1))
f.close()