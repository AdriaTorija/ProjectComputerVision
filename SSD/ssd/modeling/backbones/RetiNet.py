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
		# add lateral layers
        in_channels = [512, 256, 128, 64]
        self.lateral_layer0 = nn.Conv2d(in_channels=in_channels[0], out_channels=256, kernel_size=1, stride=1, padding=0)
        self.lateral_layer1 = nn.Conv2d(in_channels=in_channels[1], out_channels=256, kernel_size=1, stride=1, padding=0)
        self.lateral_layer2 = nn.Conv2d(in_channels=in_channels[2], out_channels=256, kernel_size=1, stride=1, padding=0)
        self.lateral_layer3 = nn.Conv2d(in_channels=in_channels[3], out_channels=256, kernel_size=1, stride=1, padding=0)
		# add smooth layers
        self.smooth_layer0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.smooth_layer1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.smooth_layer2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.smooth_layer3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
		# add downsample layer
        self.downsample_layer = nn.MaxPool2d(kernel_size=1, stride=2)
        ''' 
        self.additional_layers = nn.ModuleList([#sol
            nn.Sequential( # 16 x 128 out#sol
                nn.BatchNorm2d(output_channels[0]),#sol
                nn.LeakyReLU(0.2),#sol
                nn.Conv2d(output_channels[0], 512, kernel_size=3, padding=1),#sol
                nn.BatchNorm2d(512),#sol
                nn.LeakyReLU(0.2),#sol
                nn.Conv2d(512, output_channels[1], kernel_size=3, padding=1, stride=2),#sol
            ),#sol
            nn.Sequential( # 8x64 out#sol
                nn.BatchNorm2d(output_channels[1]),#sol
                nn.LeakyReLU(0.2),#sol
                nn.Conv2d(output_channels[1], 256, kernel_size=3, padding=1),#sol
                nn.BatchNorm2d(256),#sol
                nn.LeakyReLU(0.2),#sol
                nn.Conv2d(256, output_channels[2], kernel_size=3, padding=1, stride=2),#sol
            ),#sol
            nn.Sequential( # 4 x 32 out#sol
                nn.BatchNorm2d(output_channels[2]),#sol
                nn.LeakyReLU(0.2),#sol
                nn.Conv2d(output_channels[2], 128, kernel_size=3, padding=1),#sol
                nn.BatchNorm2d(128),#sol
                nn.LeakyReLU(0.2),#sol
                nn.Conv2d(128, output_channels[3], kernel_size=3, padding=1, stride=2),#sol
            ),#sol
            nn.Sequential( # 2 x 16 out#sol
                nn.BatchNorm2d(output_channels[3]),#sol
                nn.LeakyReLU(0.2),#sol
                nn.Conv2d(output_channels[3], 128, kernel_size=3, padding=1),#sol
                nn.BatchNorm2d(128),#sol
                nn.LeakyReLU(0.2),#sol
                nn.Conv2d(128, output_channels[4], kernel_size=3, stride=2, padding=1),#sol
            ),#sol
            nn.Sequential( # 1 x 8 out#sol
                nn.BatchNorm2d(output_channels[4]),#sol
                nn.LeakyReLU(0.2),#sol
                nn.Conv2d(output_channels[4], 128, kernel_size=2, padding=1),#sol
                nn.BatchNorm2d(128),#sol
                nn.LeakyReLU(0.2),#sol
                nn.Conv2d(128, output_channels[5], kernel_size=2,stride=2),#sol
            ),#sol
            
        ])#sol
        '''
        '''forward'''
    def forward(self, x):
		# bottom-up
        c1 = self.base_layer0(x)
        c2 = self.base_layer1(c1)
        c3 = self.base_layer2(c2)
        c4 = self.base_layer3(c3)
        c5 = self.base_layer4(c4)
		# top-down
        p5 = self.lateral_layer0(c5)
        p4 = self.upsampleAdd(p5, self.lateral_layer1(c4))
        p3 = self.upsampleAdd(p4, self.lateral_layer2(c3))
        p2 = self.upsampleAdd(p3, self.lateral_layer3(c2))
		# obtain fpn features
        p5 = self.smooth_layer0(p5)
        p4 = self.smooth_layer1(p4)
        p3 = self.smooth_layer2(p3)
        p2 = self.smooth_layer3(p2)
        p6 = self.downsample_layer(p5)
		# return all feature pyramid levels
        
        out_features = []
        out_features.append(p2)
        out_features.append(p3)
        out_features.append(p4)
        out_features.append(p5)
        out_features.append(p6)
      
        
        
        
        
        
        for out in out_features:
            print(np.shape(out))
        print("GOOD FEATURES \n")
        
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            print(expected_shape)
            
        #for additional_layer in self.additional_layers.children():#sol
        #    x = additional_layer(c5)#sol
        #   out_features.append(x)#sol


        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"


        return [p2, p3, p4, p5, p6]
        '''upsample and add'''
    def upsampleAdd(self, p, c):
        _, _, H, W = c.size()
        return F.interpolate(p, size=(H, W), mode='nearest') + c
