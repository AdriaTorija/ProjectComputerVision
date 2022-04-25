from torch import nn
from typing import Tuple, List


class BasicModel(nn.Module):
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
        
        #Define the backbone
        self.feature_extractor = nn.Sequential(#sol
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),#sol
            nn.BatchNorm2d(32),#sol
            nn.LeakyReLU(0.2),#sol
            nn.Conv2d(32, 64, kernel_size=3, padding=1),#sol
            nn.BatchNorm2d(64),#sol
            nn.LeakyReLU(0.2),#sol
            #sol
            nn.MaxPool2d(2,2), # 64x512 out#sol
            nn.Conv2d(64, 128, kernel_size=3, padding=1),#sol
            nn.BatchNorm2d(128),#sol
            nn.LeakyReLU(0.2),#sol
            nn.Conv2d(128, 128, kernel_size=3, padding=1),#sol
            nn.BatchNorm2d(128),#sol
            nn.LeakyReLU(0.2),#sol
            nn.MaxPool2d(2,2), # 32 x 256 out#sol
            nn.Conv2d(128, 256, kernel_size=3, padding=1),#sol
            nn.BatchNorm2d(256),#sol
            nn.LeakyReLU(0.2),#sol
            nn.Conv2d(256, 512, kernel_size=3, padding=1),#sol
            nn.BatchNorm2d(512),#sol
            nn.LeakyReLU(0.2),#sol
            #Ouput 32#
            nn.Conv2d(512, output_channels[0], kernel_size=3, padding=1),#so    
        ) #sol
        
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


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        x = self.feature_extractor(x)#sol
        out_features.append(x)#sol
        
        for additional_layer in self.additional_layers.children():#sol
            x = additional_layer(x)#sol
            out_features.append(x)#sol
            
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

