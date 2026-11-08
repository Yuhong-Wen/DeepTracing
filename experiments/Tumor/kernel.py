import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math


class CauchyKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(CauchyKernel, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y):
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
        d = sq_norms_x + sq_norms_y - 2. * dotprods
        if self.fixed_scale:
            res = 1/(1+d/self.scale)
        else:
            res = 1/(1+d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def forward_diag(self, x, y):
        d = ((x-y)**2).sum(dim=1)
        if self.fixed_scale:
            res = 1/(1+d/self.scale)
        else:
            res = 1/(1+d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def print_scale(self):
        print(self.scale)


class Kernel(nn.Module):
    def __init__(self, distance: np.array, scale, dtype=torch.float32, device="cpu"):
        super(Kernel, self).__init__()
        self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device))
        self.register_buffer("distance", torch.from_numpy(distance).to(device))

    def forward(self, x=None, y=None, diag_only=False):
        if diag_only:
            assert x.shape[0] == y.shape[0]
            dist_diag = self.distance[x.to(torch.long), y.to(torch.long)]
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

