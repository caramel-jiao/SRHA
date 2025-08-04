"""
Loss.py
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from config import cfg
import ipdb

def get_loss(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """
    if args.cls_wt_loss:
        ce_weight = torch.Tensor([0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                                    1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                                    1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
    else:
        ce_weight = None


    print("standard cross entropy")
    criterion = nn.CrossEntropyLoss(weight=ce_weight, reduction='mean',
                                       ignore_index=datasets.ignore_label).cuda()

    criterion_val = nn.CrossEntropyLoss(reduction='mean',
                                       ignore_index=datasets.ignore_label).cuda()
    return criterion, criterion_val

def get_loss_aux(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """
    if args.cls_wt_loss:
        ce_weight = torch.Tensor([0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                                1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                                1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
    else:
        ce_weight = None

    print("standard cross entropy")
    criterion = nn.CrossEntropyLoss(weight=ce_weight, reduction='mean',
                                    ignore_index=datasets.ignore_label).cuda()

    return criterion

def cal_local_align(feat_imnet, feat):
    '''
    gt B,H,W
    feat_imnet  B,C,H,W
    feat  B,C,H,W
    '''
    feat_diff = F.mse_loss(feat, feat_imnet, reduction='none')
    local_align = torch.mean(feat_diff)
    
    if torch.isnan(local_align):
        print(torch.isnan(feat).sum(), feat.shape)
        raise
        ipdb.set_trace()

    return local_align

def get_feat_mean_rate(x: torch.Tensor, gt: torch.Tensor, style_idx: torch.Tensor):
    '''
    x:          torch.FloatTensor,  [B, C, H, W]
    gt:         torch.LongTensor,   [B, H, W]
    style_idx:  torch.LongTensor,   [B, 1, 1]
    return: feat_mean [B, C]
    '''
    style_mask = (gt == style_idx).unsqueeze(1)
    # flatten
    x_flatten = x.reshape(*x.shape[: 2], -1)
    style_mask_flatten = style_mask.reshape(*style_mask.shape[: 2], -1)
    # mean
    feat_sum = (x_flatten * style_mask_flatten).sum(dim=-1)
    feat_num = style_mask_flatten.float().sum(dim=-1)
    feat_mean = feat_sum / feat_num                 # [B, C]
    feat_rate = feat_num / x_flatten.shape[-1]      # [B, 1]
    return feat_mean, feat_rate

def cal_semantic_regional_align(gt, feat_imnet, feat):
    imnet_mean = []
    style_mean = []
    rate_list = []
    for idx in range(gt.shape[0]):
        valid_class = gt[idx].unique()
        for class_idx in valid_class:
            imnet_itm_mean, imnet_itm_rate = get_feat_mean_rate(feat_imnet[idx][None, ...], gt[idx][None, ...], 
                    torch.tensor([[[class_idx.item()]]], device=gt.device).long())
            style_itm_mean, _ = get_feat_mean_rate(feat[idx][None, ...], gt[idx][None, ...], 
                    torch.tensor([[[class_idx.item()]]], device=gt.device).long())
            imnet_mean.append(imnet_itm_mean)
            style_mean.append(style_itm_mean)
            rate_list.append(imnet_itm_rate)
    if len(imnet_mean) > 0:
        imnet_mean = torch.cat(imnet_mean, dim=0)
        style_mean = torch.cat(style_mean, dim=0)
        rate_list = torch.cat(rate_list, dim=0)
        loss = (((style_mean - imnet_mean) ** 2) * rate_list).mean()
    else:
        loss = torch.tensor(0., device=gt.device)
        
    return loss


def calc_js_div_loss(im_prob, aug_prob):
    p_mixture = torch.clamp((aug_prob + im_prob) / 2., 1e-7, 1).log()
    consistency_loss = (
                F.kl_div(p_mixture, aug_prob, reduction='batchmean') +
                F.kl_div(p_mixture, im_prob, reduction='batchmean') 
                ) / 2.
    return consistency_loss

