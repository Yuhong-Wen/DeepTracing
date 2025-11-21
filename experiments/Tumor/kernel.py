import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math


class Kernel(nn.Module):
    def __init__(self, distance: np.array, scale, dtype=torch.float32, device="cpu"):
        super(Kernel, self).__init__()
        self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device))
        self.register_buffer("distance", torch.from_numpy(distance).to(device))

    def forward(self, x=None, y=None, diag_only=False):
        """
        Return K (x, y), if diag_only=True, only diagonal elements are returned.
        """
        if diag_only:
            assert x.shape[0] == y.shape[0], "if diag_only=True, x and y must have the same length"
            dist_diag = self.distance[x.to(torch.long), y.to(torch.long)]  # 取对角线距离
            return torch.exp(-dist_diag / torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        else:
            distance = self.distance[x.to(torch.long)][:, y.to(torch.long)]
            return torch.exp(-distance / torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))

class Cauchy_Kernel(nn.Module):
    def   __init__(self,distance:np.array, scale,fixed_scale=None, dtype=torch.float32, device="cpu"):
        super(Cauchy_Kernel, self).__init__()
        self.fixed_scale = fixed_scale
        self.register_buffer("distance", torch.from_numpy(distance).to(device))
        if fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)


    def forward(self, x=None, y=None):
        distance_matrix = self.distance[x.to(torch.long)][:, y.to(torch.long)]
        res = 1 / (1 + distance_matrix / torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

