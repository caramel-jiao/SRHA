"""
# Code Adapted from:
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#
# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import network.mynn as mynn


__all__ = ['ResNet', 'resnet50', 'resnet101']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    """
    Bottleneck Layer for Resnet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = mynn.Norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = mynn.Norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = mynn.Norm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    Resnet Global Module for Initialization
    """

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        # self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
    
        self.bn1 = mynn.Norm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                #  or isinstance(m, nn.InstanceNorm2d)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                mynn.Norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_tuple = self.layer1([x, w_arr])  # 400
        low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)  # 100
        x_tuple = self.layer3(x_tuple)  # 100
        aux_out = x_tuple[0]
        x_tuple = self.layer4(x_tuple)  # 100

        x = x_tuple[0]
        w_arr = x_tuple[1]

        return x


def load_from_depth_est(ckpt_path):
    state_dict = torch.load(ckpt_path)['encoder']
    update_dict = {}
    for key, value in state_dict.items():
        update_dict[key.replace('encoder.', '')] = value
    return update_dict


# def load_from_simclr(ckpt_path):
#     simclr_dict = torch.load(ckpt_path)['resnet']
#     resnet_dict = {}
#     # layer wise
#     rep_keys = {'net.0.0':    'conv1', 
#                       'net.0.1.0':  'bn1',}
#     for layer in range(1, 5):
#         for block in range(6):
#             simclr_rep_name = 'net.%d.blocks.%d' % (layer, block)
#             resnet_rep_name = 'layer%d.%d' % (layer, block)
#             # rep_keys[simclr_name] = resnet_name
#             add_names = {}
#             total_add_num = max([int(s_n.split('.')[5]) for s_n in simclr_dict.keys() if simclr_rep_name + '.net' in s_n])
#             if block == 0:
#                 add_names['.projection.shortcut'] = '.conv1'
#                 add_names['.projection.bn'] = '.bn1'
#                 for idx in range((total_add_num // 2) + 1):
#                     add_names['']


def resnet50(pretrained=True, args=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        if getattr(args, 'depth_est_teacher', None) is not None:
            print("########### DepthEst pretrained ##############")
            dep_state_dict = load_from_depth_est(args.depth_est_teacher)
            mynn.forgiving_state_restore(model, dep_state_dict)
        elif getattr(args, 'simclr_path', None) is not None:
            print("########### SimCLR pretrained ##############")
            simclr_state_dict = torch.load(args.simclr_path)['state_dict']
            mynn.forgiving_state_restore(model, simclr_state_dict)
        else:
            print("########### pretrained ##############")
            mynn.forgiving_state_restore(model, model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=True, args=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        if getattr(args, 'depth_est_teacher', None) is None:
            print("########### pretrained ##############")
            mynn.forgiving_state_restore(model, model_zoo.load_url(model_urls['resnet101']))
        else:
            print("########### DepthEst pretrained ##############")
            dep_state_dict = load_from_depth_est(args.depth_est_teacher)
            mynn.forgiving_state_restore(model, dep_state_dict)
    return model
