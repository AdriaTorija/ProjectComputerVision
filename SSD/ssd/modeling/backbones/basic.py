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
        module1 = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(
                kernel_size=2, 
                stride=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels= output_channels[0],
                kernel_size=3,
                stride=2,
                padding=1
            ),
        )
        self.add_module("module1", module1)
        
        module2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=output_channels[0],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels= output_channels[1],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.add_module("module2", module2)
        
        module3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels= output_channels[1],
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels= output_channels[2],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.add_module("module3", module3)
        
        module4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels= output_channels[2],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels= output_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.add_module("module4", module4)
        
        module5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels= output_channels[3],
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels= output_channels[4],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.add_module("module5", module5)
        
        module6 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels= output_channels[4],
                out_channels=128,
                kernel_size=2,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels= output_channels[5],
                kernel_size=2,
                stride=2,
                padding=0
            )
        )
        self.add_module("module6", module6)
        
  

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
        for module in self.children():
            x = module(x)
            print(x.size())
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

