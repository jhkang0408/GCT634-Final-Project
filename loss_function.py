import torch
from torch import nn
from torch.nn import functional as F

def loss_function(input, output, mean, logvar):
    recons_loss =F.mse_loss(output, input)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim = 1), dim = 0)
    return recons_loss + kld_loss