import torch.nn as nn
import torch

# class EncodingBlock(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         # Add the modules for this block
#         self.encode = nn.Sequential(
#             nn.MaxPool3d(kernal_size = 2, stride=2),
#             nn.Conv3d(self, in_channel)
#         )
#         # Add the string of encoder modules to the class
#         self.add_module("encoding_block", self.encode)


class UNet_3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        starting_features = 64,
        feature_multiplication = 2,
        network_depth = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.starting_features = starting_features
        self.network_depth = network_depth

        self.input_layer = nn.Sequential(
            nn.Conv3d(in_channels, starting_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv3d(starting_features, starting_features, kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )

        encoder = []
        current_depth = 0
        for _ in range(self.network_depth):
            encoder.append(
                nn.Sequential(
                    nn.MaxPool3d(kernel_size=2, stride=2),
                    # in 64, out 128
                    # in 128, out 128
                    nn.Conv3d(starting_features * feature_multiplication ** current_depth, starting_features * feature_multiplication ** (current_depth+1), kernel_size=3, padding=1, bias=True),
                    nn.ReLU(),
                    nn.Conv3d(starting_features * feature_multiplication ** (current_depth+1), starting_features * feature_multiplication ** (current_depth+1), kernel_size=3, padding=1, bias=True),
                    nn.ReLU()
                )
            )
            current_depth += 1

        self.encoder = nn.ModuleList(encoder)

        self.bottom_layer = nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                nn.Conv3d(starting_features * feature_multiplication ** current_depth, starting_features * feature_multiplication ** (current_depth+1), kernel_size=3, padding=1, bias=True),
                nn.ReLU(),
                nn.Conv3d(starting_features * feature_multiplication ** (current_depth+1), starting_features * feature_multiplication ** (current_depth+1), kernel_size=3, padding=1, bias=True),
                nn.ReLU(),
                nn.ConvTranspose3d(starting_features * feature_multiplication ** (current_depth + 1), starting_features * feature_multiplication ** current_depth, kernel_size=2, stride=2, bias=True)
        )

        current_depth += 1
        
        decoder = []
        # One less since the final encoding layer will be the "bottom" of the UNet
        for _ in range(self.network_depth):
            decoder.append(
                nn.Sequential(
                    nn.Conv3d(starting_features * feature_multiplication ** current_depth, starting_features * feature_multiplication ** (current_depth-1), kernel_size=3, padding=1, bias=True),
                    nn.ReLU(),
                    nn.Conv3d(starting_features * feature_multiplication ** (current_depth-1), starting_features * feature_multiplication ** (current_depth-1), kernel_size=3, padding=1, bias=True),
                    nn.ReLU(),
                    nn.ConvTranspose3d(starting_features * feature_multiplication ** (current_depth-1), starting_features * feature_multiplication ** (current_depth-2), kernel_size=2, stride=2, bias=True)      
                )
            )
            current_depth -= 1
        current_depth -= 1

        self.decoder = nn.ModuleList(decoder)

        self.output_layer = nn.Sequential(
            nn.Conv3d(starting_features * feature_multiplication ** (current_depth+1), starting_features * feature_multiplication ** current_depth, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv3d(starting_features * feature_multiplication ** current_depth, starting_features * feature_multiplication ** current_depth, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv3d(starting_features * feature_multiplication ** current_depth, out_channels, kernel_size=1, bias=True)
            )

    def forward(self, x):
        x_features = self.input_layer(x)

        # Hold features in list
        x_stack = [x_features]
        for layer in self.encoder:
            x_features = layer(x_features)
            x_stack.append(x_features)
        
        for layer in self.decoder:
            x_features = layer(torch.cat((x_features, x_stack.pop()), dim=1))

        x_features = self.output_layer(torch.cat((x_features, x_stack.pop()), dim=1))

        return x_features



    # class DoubleConv(nn.Sequential):
#     """2(conv+LeakyRelU)"""
#     def __init__(
#         self,
#         in_channels,
#         out_channels
#     ):
#         super().__init__()

#         self.double_conv = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv3d(out_channels, out_channels, kernel_size=3),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         )
#         self.add_module("double_conv", self.double_conv)


# class Down(nn.Sequential):
#     def __init__(
#         self,
#         in_channels,
#         out_channels
#     ):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             # nn.MaxPool3d(kernel_size=2),
#             DoubleConv(in_channels, out_channels)
#         )
#         self.add_module("maxpool_conv", self.maxpool_conv)


# class UpCat(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         cat_channels,
#         out_channels
#     ):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=1),
#             nn.BatchNorm3d(out_channels),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             # nn.PixelShuffle(2),
#             nn.BatchNorm3d(out_channels),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         )
#     def forward(self, x1, x2):
#         x_1 = self.maxpool_conv(x1)

#         x = self.maxpool_conv(torch.cat((x2, x_1)), dim=1)

#         return x

# class Up(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.up = nn.Sequential(
#             nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
#             # self.norm(in_channels),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
#             # self.norm(in_channels),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv3d(in_channels, out_channels, kernel_size=1)
#             )
#         self.add_module("up", self.up)


# class UNet3D(nn.Module):
#     def __init__(
#         self,
#         patch_size,
#         in_channels,
#         out_channels,
#         feature_map = (16, 32, 64, 128, 256),
#         output_activation="sigmoid",
#         normailisation_method=None,
#         **kwargs
#     ):
#         self.patch_size = patch_size
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.feature_map = feature_map
#         self.output_activation = output_activation
#         self.normailisation_method = normailisation_method

#         # if self.normailisation_method is None:
#         #     self.norm = nn.Identity
#         # elif self.normailisation_method == "batch":
#         #     self.norm = nn.BatchNorm3D
#         # else:
#         #     raise ValueError("Implement more norm methods please")

#         super().__init__()
#         self.in_conv = nn.Sequential(
#             DoubleConv(self.in_channels, self.feature_map[0])
#         )
#         self.down_1 = Down(self.feature_map[0], self.feature_map[1])
#         self.down_2 = Down(self.feature_map[1], self.feature_map[2])
#         self.down_3 = Down(self.feature_map[2], self.feature_map[3])
#         self.down_4 = Down(self.feature_map[3], self.feature_map[4])

#         self.up_4 = Up(self.feature_map[4], self.feature_map[3])
#         self.up_3 = Up(self.feature_map[3], self.feature_map[2])
#         self.up_2 = Up(self.feature_map[2], self.feature_map[1])
#         self.up_1 = Up(self.feature_map[1], self.feature_map[0])

#         self.final_conv = nn.Conv3d(feature_map[0], out_channels, kernel_size=1)

#     def forward(self, x):
#         x0 = self.in_conv(x)
#         x1 = self.down_1(x0)
#         x2 = self.down_2(x1)
#         x3 = self.down_3(x2)
#         x4 = self.down_4(x3)

#         u4 = self.up_4(x4)
#         u4 = torch.cat((u4,x3),1)
#         u3 = self.up_3(u4)
#         u3 = torch.cat((u3,x2),1)
#         u2 = self.up_2(u3)
#         u2 = torch.cat((u2,x1),1)
#         u1 = self.up_1(u2)
        
#         logits = self.final_conv(u1)

#         return logits