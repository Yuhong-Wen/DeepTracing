import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Kernel(nn.Module):
    def __init__(self,distance:np.array, scale,fixed_scale=None, dtype=torch.float32, device="cpu"):
        super(Kernel, self).__init__()
        self.fixed_scale = fixed_scale
        self.register_buffer("distance", torch.from_numpy(distance).to(device))
        if fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x=None, y=None):
        distance_matrix = self.distance[x.to(torch.long)][:, y.to(torch.long)]
        return torch.exp(-distance_matrix/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))


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


