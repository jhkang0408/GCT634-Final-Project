import torch
from torch import nn
from torch.nn import functional as F

def loss_function(input, output, mean, logvar):
    #recons_loss =F.mse_loss(output, input)
    recons_loss =F.l1_loss(output, input)
    #print(output.view(1,-1).shape)
    #print(output.view(1,-1)[:5])
    #print(input.view(1,-1).shape)
    #print(input.view(1,-1)[:5])

    #recons_loss = F.binary_cross_entropy_with_logits(output.view(1,-1), input.view(1,-1), reduction='mean')
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim = 1), dim = 0)
    return recons_loss + kld_loss