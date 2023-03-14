import torch
x=torch.rand(2,18)
print(x)


b, c= x.shape
x = x.reshape(b, 3, -1)
#print(x)
#print(x.shape)
x = x.permute(0, 2, 1)
#print(x)
#print(x.shape)
# flatten
x = x.reshape(b, -1)
print(x)
#print(x.shape)
