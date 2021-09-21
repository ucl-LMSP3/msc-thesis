import numpy as np
import torch


def init_weights(net):
    if type(net) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(net.weight)
        net.bias.data.fill_(0.01)


def _zero_mean_function(x):
    return 0


def get_mean_params(gp):
    params = []
    for name, param in gp.named_parameters():
        if 'mean_function.mean_fn' in name:
            params.append(param)
    return params


def get_kernel_params(gp):
    params = []
    for name, param in gp.named_parameters():
        if 'mean_function.mean_fn' in name:
            continue
        params.append(param)
    return params

