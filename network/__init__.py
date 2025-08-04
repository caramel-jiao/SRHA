"""
Network Initializations
"""

import logging
import importlib
import torch
from torch import nn 
import torch.nn.functional as F
import datasets


def get_net(args):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(args=args, num_classes=datasets.num_classes)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.3f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net


def warp_network_in_dataparallel(net, gpuid):
    """
    Wrap the network in Dataparallel
    """
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpuid], find_unused_parameters=True)
    return net


def get_model(args, num_classes):
    """
    Fetch Network Function Pointer
    """
    network = args.arch
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(args=args, num_classes=num_classes)
    return net


def _get_module(model):
    if isinstance(model, torch.nn.Module):
        return model
    else:
        return model.module


def init_ema_weights(model, ema_model):
    for param in _get_module(ema_model).parameters():
        param.detach_()
    mp  = list(_get_module(model).parameters())
    mcp = list(_get_module(ema_model).parameters())
    for i in range(0, len(mp)):
        if not mcp[i].data.shape:  # scalar tensor
            mcp[i].data = mp[i].data.clone()
        else:
            mcp[i].data[:] = mp[i].data[:].clone()
            

def update_ema(model, ema_model, iter, alpha):
    alpha_teacher = min(1 - 1 / (iter + 1), alpha)
    for ema_param, param in zip(_get_module(ema_model).parameters(),
                                _get_module(model).parameters()):
        if not param.data.shape:  # scalar tensor
            ema_param.data = \
                alpha_teacher * ema_param.data + \
                (1 - alpha_teacher) * param.data
        else:
            ema_param.data[:] = \
                alpha_teacher * ema_param[:].data[:] + \
                (1 - alpha_teacher) * param[:].data[:]


def freeze_dropout_ema(ema_model):
    for param in _get_module(ema_model).parameters():
        if isinstance(param, torch.nn.modules.dropout._DropoutNd):
            param.training = False


def get_gradient(segm, edge_width):
    '''
    segm: [N, class, H, W]
    return: [N, 2*class, H + 1, W + 1], grad on v and h direction
    '''
    base = F.pad(segm, [edge_width, 0, edge_width, 0])
    grad_h = F.pad(segm, [0, edge_width, edge_width, 0]) - base
    grad_v = F.pad(segm, [edge_width, 0, 0, edge_width]) - base
    grad = torch.cat([grad_h, grad_v], dim=1)
    return grad
