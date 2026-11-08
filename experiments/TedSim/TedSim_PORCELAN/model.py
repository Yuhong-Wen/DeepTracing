from torch import nn
import torch

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, enc_layers, dec_layers=None, non_linearity=nn.GELU()):
        super(AutoEncoder, self).__init__()
        if dec_layers is None:
            dec_layers = enc_layers
        self.enc = nn.Sequential(*([nn.Linear(input_dim, hidden_dim)] + (enc_layers - 1) * [non_linearity,
                                                                                            nn.Linear(hidden_dim,
                                                                                                      hidden_dim)]))
        self.dec = nn.Sequential(
            *((dec_layers - 1) * [non_linearity, nn.Linear(hidden_dim, hidden_dim)]) + [non_linearity,
                                                                                        nn.Linear(hidden_dim,
                                                                                                  input_dim)])

    def forward(self, x, return_z=False):
        z = self.enc(x)
        if return_z:
            return self.dec(z), z
        return self.dec(z)