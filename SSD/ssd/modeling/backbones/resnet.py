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
        self.resnet = resnet50(pretrained=True)
    
        module1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[2],
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.output_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        module2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[3],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=self.output_channels[4],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        module3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[4],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=self.output_channels[5],
                kernel_size=3,
                stride=1,
                padding=0
            )
        )
        self.net = nn.ModuleList([module1, module2, module3])
        
        
        
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
        for module in self.resnet.children():
            if isinstance(module, torch.nn.AdaptiveAvgPool2d):  # Don't use the last linear layers
                break
            x = module(x)
            out_features.append(x)

        # Only use 3 outputs
        out_features = out_features[-3:]
        
         for cmodule in self.net:
            x = cmodule(x)
            out_features.append(x)
            
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

