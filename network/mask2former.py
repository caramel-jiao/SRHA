import os 
from easydict import EasyDict 
import yaml 

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as tdist
from network import Resnet, Mobilenet, Shufflenet
from network.mynn import initialize_weights, Norm2d, Upsample

from detectron2.layers import ShapeSpec
from detectron2.modeling.postprocessing import sem_seg_postprocess

from .mask2former_head.mask_former_head import MaskFormerHead
from .mask2former_head.matcher import HungarianMatcher
from .mask2former_head.criterion import SetCriterion
from .deepv3 import SemanticRearrangementModule


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return EasyDict(data)


def get_r50_m2f_decoder(channel_list, stride_list):
    cfg = read_yaml('network/mask2former_head/config/Base-Cityscapes-SemanticSegmentation.yaml')
    m2f_cfg = read_yaml('network/mask2former_head/config/maskformer2_R50_bs16_90k.yaml')
    cfg.update(m2f_cfg)
    
    shape_list = [ShapeSpec() for _ in range(4)]
    # channel_list = [256, 512, 1024, 2048]
    # stride_list = [1, 2, 2, 2]
    for idx in range(4):
        shape_list[idx].channels = channel_list[idx]
        shape_list[idx].stride = stride_list[idx]
    
    input_shape = {
        k: v for k, v in zip(
            ["res2", "res3", "res4", "res5"], 
            shape_list
        )
    }
    
    m2f_decoder = MaskFormerHead.from_config(cfg, input_shape)
    m2f_decoder = MaskFormerHead(**m2f_decoder)
    return m2f_decoder, cfg
    

class Mask2Former(nn.Module):
    """
    Implement DeepLab-V3 model
    """

    def __init__(self, num_classes, trunk='resnet-50', args=None, 
                 variant=None):
        super(Mask2Former, self).__init__()
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

        self.m2f_decoder, cfg = get_r50_m2f_decoder(
            [channel_3rd, channel_4th, prev_final_channel, final_channel],
            [1, 2, 2, 2]
        )
        
        # criterion
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "masks"]
        
        criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )
        self.criterion = criterion


    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]            # [100, 20] -> [100, 19]
        mask_pred = mask_pred.sigmoid()                             # [100, 192, 192], 0~1
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)   # [19, 100] @ [100, 192, 192]
        return semseg
    

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

        dec_feature = {
            'res2': f_style['layer1'],
            'res3': f_style['layer2'],
            'res4': f_style['layer3'],
            'res5': f_style['layer4'],
        }
        m2f_outputs = self.m2f_decoder(dec_feature)
        '''
        ['pred_logits', 'pred_masks', 'aux_outputs']                                                                                                           
        [('pred_logits', torch.Size([6, 100, 20])), ('pred_masks', torch.Size([6, 100, 192, 192]))]
        [('aux_outputs', 9)] 
        '''
        
        mask_cls_results = m2f_outputs["pred_logits"]
        mask_pred_results = m2f_outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(x_size[-2], x_size[-1]),
            mode="bilinear",
            align_corners=False,
        )

        processed_results = []
        image_size = (x_size[-2], x_size[-1])
        for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
            height = image_size[0]
            width = image_size[1]
            processed_results.append({})

            # false for semantic seg
            self.sem_seg_postprocess_before_inference = False

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = (sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            r = (self.semantic_inference)(mask_cls_result, mask_pred_result)
            if not self.sem_seg_postprocess_before_inference:
                r = (sem_seg_postprocess)(r, image_size, height, width)
            processed_results[-1]["sem_seg"] = r
            
        main_out = torch.stack([pr["sem_seg"] for pr in processed_results], dim=0)
                    
        
        if self.training or out_prob:
            output = {
                'main_out': main_out,
                'm2f_outputs': m2f_outputs,
                'aux_out': None,
                'features': f_style,
                'rand_info': rand_info
            }
                
            return output
        else:
            del m2f_outputs
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



def R50M2F(args, num_classes):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return Mask2Former(num_classes, trunk='resnet-50', args=args)

