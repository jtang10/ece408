import numpy as np
import torch

# x: B C H W: 2 3 7 7
# k: M C K K: 4 3 3 3
# y: B M Ho Wo: 2 4 5 5
x = np.arange(1, 148).reshape(1, 3, 7, -1)
x = np.concatenate((x, x), axis=0)
x = torch.from_numpy(x)
k = np.ones(108).reshape((4,3,3,3))
k = torch.from_numpy(k)

print(x.shape)
print(k.shape)