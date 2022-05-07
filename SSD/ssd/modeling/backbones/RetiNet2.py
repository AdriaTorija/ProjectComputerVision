from torch import nn
import torchvision
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np



class FPNResNets2(nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]


img_size   128, 1024                                               300,300      
    NEW                                                     OLD
    [shape(-1, output_channels[0], 32, 256),                38, 38)
     shape(-1, output_channels[1], 16, 128),                19, 19),
     shape(-1, output_channels[2], 8, 64),                  10, 10),            
     shape(-1, output_channels[3], 4, 32),                  5, 5)
     shape(-1, output_channels[3], 2, 16),                  3, 3)
     shape(-1, output_channels[4], 1, 8)]                   1, 1)
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        self.backbone = torchvision.models.resnet101(pretrained=True)
        self.fpn=torchvision.ops.FeaturePyramidNetwork([64,128,256,512,1024],256)
        self.out_channels = [256 for i in range(6)]

		# parse backbone
        self.backbone.layer0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu,self.backbone.maxpool)
        self.backbone.layer5= nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(1024), self.backbone.relu)
        self.backbone.layer6= nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), self.backbone.relu)

        '''forward'''
    def forward(self, x):
		# bottom-up
        c1 = self.backbone.layer0(x)
        c2 = self.backbone.layer1(c1)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        print(np.shape(c5))
        c6= self.backbone.layer5(c5)
        print(np.shape(c6))
        c7= self.backbone.layer6(c6)
        print(np.shape(c7))
        
        from collections import OrderedDict
        d = OrderedDict()
        d['feat0'] = c2
        d['feat1'] = c3
        d['feat2'] = c4
        d['feat3'] = c5
        d['feat4'] = c6
        d['feat5'] = c7
        
        out_features = list(self.fpn(d).values())

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)   
