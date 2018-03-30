import torch
import torch.nn as nn
from scipy import ndimage as ndi
import numpy as np

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

def initializeBilinear(tensor):
    size = tensor.size()
    matrix = np.ones(size)
    matrix[int(size[0]/2-0.5), int(size[1]/2-0.5)] = 0
    dist = ndi.distance_transform_edt(matrix)+1
    dist = 1/dist
    return torch.Tensor(dist)