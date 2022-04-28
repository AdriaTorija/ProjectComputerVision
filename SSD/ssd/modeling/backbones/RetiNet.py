'''
Function:
	Feature Pyramid Network of ResNets
Author:
	Charles
'''
from torch import nn
import torchvision
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np

'''FPN by using ResNets'''
class FPNResNets(nn.Module):
    def __init__(self,output_channels: List[int],image_channels: int,output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        self.backbone = torchvision.models.resnet34(pretrained=True)
        
		# parse backbone
        self.base_layer0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.base_layer1 = nn.Sequential(self.backbone.layer1)
        self.base_layer2 = nn.Sequential(self.backbone.layer2)
        self.base_layer3 = nn.Sequential(self.backbone.layer3)
        self.base_layer4 = nn.Sequential(self.backbone.layer4)
		
        
        '''forward'''
    def forward(self, x):
		# bottom-up
        c1 = self.base_layer0(x)
        c2 = self.base_layer1(c1)
        c3 = self.base_layer2(c2)
        c4 = self.base_layer3(c3)
        c5 = self.base_layer4(c4)
		
        from collections import OrderedDict
        d = OrderedDict()
        d['feat0'] = c1
        d['feat1'] = c2
        d['feat2'] = c3
        d['feat3'] = c4
        d['feat4'] = c5

        m=torchvision.ops.FeaturePyramidNetwork([64,64,128,256,512],256)
        output= m(d)
        
        for k, v in out.items():
            out_features.append(v)
        self.out_channels = [256 for i in range(5)]
      
    
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"


        return tuple(out_features)   


