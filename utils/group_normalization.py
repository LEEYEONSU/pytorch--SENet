import torch
import numpy as np 
import torch.nn as nn 

# pytorch implementation for group normalization
class GroupNorm2d(nn.Module):

    def __init__(self, num_channels, num_groups = 8, eps = 1e-5):
        super(GroupNorm2d, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.gamma = nn.Parameter(torch.ones(num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):

        N, C, H, W = x.size()

        # (N, G, C//G, H, W)
        x = x.view(N, self.num_groups, self.num_channels//self.num_groups, H, W)

        # (N, G, 1, 1, 1)
        mean = torch.mean(x, dim = (2,3,4), keepdim = True)
        var = torch.var(x, dim =(2,3,4), keepdim = True)    

        # (N, G, C//G, H, W)
        x = (x - mean) /torch.sqrt( var + self.eps)
        # (N, C, H, W)
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta