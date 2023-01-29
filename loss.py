import torch
import numpy as np

def pinball_loss(true: torch.FloatTensor, pred: torch.FloatTensor):
    idx_y_big = torch.where((true - pred) > 0)
    idx_y_small = torch.where((true - pred) < 0)

    loss = torch.sum((true - pred)[idx_y_big]) + torch.sum((pred - true)[idx_y_small])

    return loss
