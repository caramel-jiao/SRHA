"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as tdist
from network import Resnet, Mobilenet, Shufflenet
from network.mynn import initialize_weights, Norm2d, Upsample


def get_class_mean_std(x: torch.Tensor, gt: torch.Tensor, style_idx: torch.Tensor):
    '''
    x:          torch.FloatTensor,  [B, C, H, W]
    gt:         torch.LongTensor,   [B, H, W]
    style_idx:  torch.LongTensor,   [B, 1, 1]
    return: feat_mean [B, C] , feat_std [B, C], style_mask [B, H, W]
    '''
    style_mask = (gt == style_idx).unsqueeze(1)
    # flatten
    x_flatten = x.reshape(*x.shape[: 2], -1)
    style_mask_flatten = style_mask.reshape(*style_mask.shape[: 2], -1)
    # mean
    feat_sum = (x_flatten * style_mask_flatten).sum(dim=-1)
    feat_num = style_mask_flatten.float().sum(dim=-1)
    feat_mean = feat_sum / feat_num                 # [B, C]
    # std
    feat_var = ( ( (x_flatten - feat_mean.unsqueeze(-1)) * style_mask_flatten ) ** 2 ).sum(dim=-1) / feat_num
    feat_std = feat_var ** 0.5 + 1e-7               # [B, C]
    return feat_mean, feat_std, style_mask.squeeze(1)


def standard_class_feature_item(x: torch.Tensor, gt: torch.Tensor, style_mean: torch.Tensor, style_std: torch.Tensor,
                                special_class=None):
    '''
    once an item. 
    x: [C, H, W] 
    gt: [H, W] 
    style_mean, style_std: [C] 
    special_class: None, or tensor(idx) 
    '''
    # res = torch.zeros_like(x)
    res = x.clone()
    if special_class is None:
        all_class = gt.unique()
    else:
        all_class = [special_class]
    for class_idx in all_class:
        class_idx = torch.tensor([[[class_idx.item()]]], device=gt.device).long()
        class_mean, class_std, class_mask = get_class_mean_std(x.unsqueeze(0), gt.unsqueeze(0), class_idx)
        class_mean, class_std, class_mask = class_mean.squeeze(0), class_std.squeeze(0), class_mask.squeeze(0)
        class_norm = (x[:, class_mask] - class_mean.unsqueeze(1)) / class_std.unsqueeze(1)      # [C, num]
        res[:, class_mask] = class_norm * style_std.unsqueeze(1) + style_mean.unsqueeze(1)
    return res


class SemanticRearrangementModule(nn.Module):
    def __init__(self, num_classes, args=None):
        super(SemanticRearrangementModule, self).__init__()
        self.num_classes = num_classes
        self.args = args
    
    def forward(self, x, gt, aug_rand_info=None):
        gt = gt.unsqueeze(1).float()
        gt = F.interpolate(gt, size=x.shape[-2: ], mode='nearest')
        gt = gt.squeeze(1).long()
        
        _rand_info = {}
        x_style = torch.zeros_like(x)
           
        x_style = x.clone()
        for idx in range(gt.shape[0]):
            # get all classes' styles
            valid_class = gt[idx].unique()
            style_mean = []
            style_std = []
            for class_idx in valid_class:
                sm, ss, _ = get_class_mean_std(x[idx].unsqueeze(0), gt[idx].unsqueeze(0), 
                                                torch.tensor([[[class_idx.item()]]], device=gt.device).long())
                style_mean.append(sm)       # [1, C]
                style_std.append(ss)        # [1, C]
            style_mean = torch.cat(style_mean, dim=0)       # [n_class, C]
            style_std = torch.cat(style_std, dim=0)         # [n_class, C]
            # get random dist
            concentration = torch.tensor([self.args.concentration_coeff] * len(valid_class), device=gt.device)
            dirichlet = tdist.dirichlet.Dirichlet(concentration=concentration)
            for class_idx in valid_class:
                # random dirichlet sample
                if aug_rand_info is None:
                    combine_weights = dirichlet.sample((1,))        # [1, n_class]
                    combine_weights = combine_weights.detach()
                else:
                    combine_weights = aug_rand_info[(idx, class_idx.item())]
                # combine styles & stylization
                sm = (combine_weights @ style_mean).squeeze(0)          # [C,]
                ss = (combine_weights @ style_std).squeeze(0)           # [C,]
                x_style[idx] = standard_class_feature_item(x_style[idx], gt[idx], sm, ss,
                                                            special_class=class_idx)
                _rand_info[(idx, class_idx.item())] = combine_weights.clone()
                 
        x_style = x_style.clone().detach().requires_grad_(True)
        return x, x_style, _rand_info


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        print("output_stride = ", output_stride)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            Norm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    """

    def __init__(self, num_classes, trunk='resnet-101', args=None, 
                 variant=None):
        super(DeepV3Plus, self).__init__()
        self.args = args
        self.trunk = trunk
        self.variant = variant

        if getattr(args, 'srm', False):
            self.SRM = SemanticRearrangementModule(num_classes, args)
        else:
            self.SRM = None
            
        channel_1st = 3
        channel_2nd = 64
        channel_3rd = 256
        channel_4th = 512
        prev_final_channel = 1024
        final_channel = 2048

        if trunk == 'resnet-50':
            resnet = Resnet.resnet50(args=args)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101':
            resnet = Resnet.resnet101(args=args)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'shufflenetv2':
            channel_1st = 3
            channel_2nd = 24
            channel_3rd = 116
            channel_4th = 232
            prev_final_channel = 464
            final_channel = 1024
            resnet = Shufflenet.shufflenet_v2_x1_0(pretrained=True)
            
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.maxpool)
            resnet.layer1 = resnet.stage2
            resnet.layer2 = resnet.stage3
            resnet.layer3 = resnet.stage4
            resnet.layer4 = resnet.conv5
            
            if self.variant == 'D':
                for n, m in resnet.layer2.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                for n, m in resnet.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif self.variant == 'D16':
                for n, m in resnet.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            else:
                print("Not using Dilation ")
                
        elif trunk == 'mobilenetv2':
            channel_1st = 3
            channel_2nd = 16
            channel_3rd = 32
            channel_4th = 64

            prev_final_channel = 320

            final_channel = 1280
            resnet = Mobilenet.mobilenet_v2(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.features[0],
                                        resnet.features[1])
            resnet.layer1 = nn.Sequential(resnet.features[2], resnet.features[3],
                                        resnet.features[4], resnet.features[5], resnet.features[6])
            resnet.layer2 = nn.Sequential(resnet.features[7], resnet.features[8], resnet.features[9], resnet.features[10])

            resnet.layer3 = nn.Sequential(resnet.features[11], resnet.features[12], resnet.features[13],
                                        resnet.features[14], resnet.features[15], resnet.features[16],
                                        resnet.features[17])
            resnet.layer4 = nn.Sequential(resnet.features[18])

            if self.variant == 'D':
                for n, m in resnet.layer2.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                for n, m in resnet.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif self.variant == 'D16':
                for n, m in resnet.layer3.named_modules():
                    if isinstance(m, nn.Conv2d) and m.stride==(2,2):
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            else:
                print("Not using Dilation ")
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if getattr(args, 'freeze_resnet', False):
            self.frozen_layers = ['layer%d' % i for i in range(5)]
        else:
            self.frozen_layers = []
            
        if getattr(args, 'freeze_layer0', False):
            self.frozen_layers.append('layer0')
            
        if getattr(args, 'freeze_layer1', False):
            self.frozen_layers.append('layer1')
            
        if getattr(args, 'freeze_layer2', False):
            self.frozen_layers.append('layer2')
            
        if getattr(args, 'freeze_layer3', False):
            self.frozen_layers.append('layer3')
            
        if getattr(args, 'freeze_layer4', False):
            self.frozen_layers.append('layer4')

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        os = 16
        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.dsn = nn.Sequential(
            nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
            Norm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        initialize_weights(self.dsn)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        self.eps = 1e-5


    def forward(self, x, out_prob=False, return_style_features=False,
                seg_gt=None, aug_rand_info=None, ):

        if isinstance(return_style_features, list) and len(return_style_features):
            multi_feature = True
        else:
            multi_feature = False

        x_size = x.size()  # 800
        
        x = self.layer0(x)

        if not multi_feature and return_style_features:
            return x
        f_style = {}

        rand_info = None
        # SRM for layer0
        if self.SRM is not None:
            rand_info = None
            if seg_gt is not None:
                if seg_gt.shape[0] == 2 * x.shape[0]:
                    seg_gt = seg_gt[: x.shape[0]]
                x, x_style, rand_info = self.SRM(x, seg_gt, aug_rand_info)
                x = torch.cat((x, x_style), dim=0)
                
        if getattr(self.args, 'style_elimination', False):    
            x = (x - x.mean([2, 3], keepdim=True)) / (x.std([2, 3], keepdim=True) + 1e-7)
           
        f_style['layer0'] = x

        x = self.layer1(x)  # 400
        low_level = x
        f_style['layer1'] = low_level

        x = self.layer2(x)  # 100
        f_style['layer2'] = x

        x = self.layer3(x)  # 100
        aux_out = x
        f_style['layer3'] = aux_out

        x = self.layer4(x)  # 100
        f_style['layer4'] = x

        # f_style_return = {}
        all_levels = list(f_style.keys())
        if multi_feature:
            for k in all_levels:
                if k not in return_style_features:
                    del f_style[k]

        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)
        
        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final1(dec0)
        dec2 = self.final2(dec1)
        main_out = Upsample(dec2, x_size[2:])
        
        if self.training or out_prob:
            aux_out = self.dsn(aux_out)
            output = {
                'main_out': main_out,
                'aux_out': aux_out,
                'features': f_style,
                'rand_info': rand_info
            }
                
            return output
        else:
            return main_out

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self



def DeepR50V3PlusD(args, num_classes):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, trunk='resnet-50', args=args)

def DeepR101V3PlusD(args, num_classes):
    """
    Resnet 101 Based Network, the origin 7x7
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3Plus(num_classes, trunk='resnet-101', args=args)

def DeepShuffleNetV3PlusD(args, num_classes):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3Plus(num_classes, trunk='shufflenetv2', variant='D16', args=args)

def DeepMobileNetV3PlusD(args, num_classes):
    """
    MobileNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : mobilenetv2")
    return DeepV3Plus(num_classes, trunk='mobilenetv2', variant='D16', args=args)
