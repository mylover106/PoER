import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ReconPostProcessor:
    @torch.no_grad()
    def __call__(self, net, data):
        """ return the ood confidence
        data: [N, 3, H, W]
        """
        z, recon = net(data)
        conf = 1 - F.mse_loss(data, recon, reduction='none').mean((1, 2, 3))
        return conf
