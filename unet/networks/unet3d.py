import torch.nn as nn
import torch

from typing import Literal


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_kernel_size=3,
        normalization: Literal[None, "group", "batch"] = "batch",
        num_groups=8,
    ):
        super().__init__()
        # Calculate padding based on kernel size
        # Thus, the decoder path does not require cropping before concatenation
        # Though, if the conv_kernel_size isn't divisible, then will throw an error
        self.padding = (conv_kernel_size - 1) // 2
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=conv_kernel_size,
            padding=self.padding,
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=conv_kernel_size,
            padding=self.padding,
        )

        # Find normalization method
        if normalization == "group":
            norm = nn.GroupNorm(num_groups, out_channels)
        elif normalization == "batch":
            norm = nn.BatchNorm3d(out_channels)
        else:
            norm = nn.Identity()
        # Create layers
        layers = []
        layers.append(self.conv1)
        layers.append(norm)
        layers.append(self.relu)
        layers.append(self.conv2)
        layers.append(norm)
        layers.append(self.relu)

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


def calculate_feature_channels(initial_feature, multiplier, network_depth):
    """Calculate progressive feature channels"""
    return [initial_feature * multiplier ** k for k in range(network_depth)]


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        feature_map: list[int],
        normalization,
        num_groups,
        dropout_probability: float,
        conv_kernel_size: int,
    ):
        super().__init__()
        self.dropout_probability = dropout_probability
        # Build the encoder features. Copy to prevent in-place insert
        self.encoder_features = feature_map.copy()
        self.encoder_features.insert(0, input_channels)
        # Build the encoder blocks from feature_map
        self.encoder_blocks = nn.ModuleList(
            [
                ConvBlock(
                    self.encoder_features[i],
                    self.encoder_features[i + 1],
                    normalization=normalization,
                    num_groups=num_groups,
                    conv_kernel_size=conv_kernel_size,
                )
                for i in range(len(self.encoder_features) - 1)
            ]
        )
        self.pool = nn.MaxPool3d(kernel_size=2)
        # During model.eval(), dropout will become an identity module (ie. do nothing)
        self.dropout = nn.Dropout3d(p=self.dropout_probability)

    def forward(self, x):
        features = []
        for block in self.encoder_blocks:
            # Run Conv
            x = block(x)
            features.append(x)
            # Contractive path
            x = self.pool(x)
            # Apply dropout
            x = self.dropout(x)
        return features


class Decoder(nn.Module):
    def __init__(
        self,
        feature_map: list[int],
        normalization,
        num_groups,
        dropout_probability: float,
        conv_kernel_size: int,
    ):
        super().__init__()
        self.dropout_probability = dropout_probability
        self.feature_map = feature_map
        self.up_conv = nn.ModuleList(
            [
                nn.ConvTranspose3d(
                    feature_map[i], feature_map[i + 1], kernel_size=2, stride=2
                )
                for i in range(len(feature_map) - 1)
            ]
        )
        self.decoder_block = nn.ModuleList(
            [
                ConvBlock(
                    feature_map[i],
                    feature_map[i + 1],
                    normalization=normalization,
                    num_groups=num_groups,
                    conv_kernel_size=conv_kernel_size,
                )
                for i in range(len(feature_map) - 1)
            ]
        )
        self.dropout = nn.Dropout3d(p=self.dropout_probability)

    def forward(self, x, encoder_features):
        for i in range(len(self.feature_map) - 1):
            # Expansive path with ConvTranspose3d
            x = self.up_conv[i](x)
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.decoder_block[i](x)
            # Apply dropout
            x = self.dropout(x)
        return x


class UNet3D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        initial_feature=64,
        feature_multiplier=2,
        network_depth=4,
        normalization: Literal[None, "group", "batch"] = "group",
        num_groups: int = 8,
        conv_kernel_size=3,
        dropout_probability=0,
        activation=None,
    ):
        super().__init__()
        self.feature_map = calculate_feature_channels(
            initial_feature, feature_multiplier, network_depth
        )
        self.encoder = Encoder(
            input_channels,
            self.feature_map,
            normalization=normalization,
            num_groups=num_groups,
            dropout_probability=dropout_probability,
            conv_kernel_size=conv_kernel_size,
        )
        self.decoder = Decoder(
            self.feature_map[::-1],
            normalization=normalization,
            num_groups=num_groups,
            dropout_probability=dropout_probability,
            conv_kernel_size=conv_kernel_size,
        )
        # Output will take the lowest decoder features and output the desired number of classess
        out_layers = []
        out_layers.append(nn.Conv3d(self.feature_map[0], num_classes, kernel_size=1))

        if activation is not None:
            if activation.casefold() == "softmax":
                out_layers.append(nn.Softmax(dim=1))
            elif activation.casefold() == "sigmoid":
                out_layers.append(nn.Sigmoid())
        
        self.head = nn.Sequential(*out_layers)

    def forward(self, x):
        # Run the encoder
        encoder_features = self.encoder(x)
        # Decoder takes the 'bottleneck' featute map with encoder_features[::-1][0]
        # Decoder also takes all encoder features above the bottleneck for concatenation
        output = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
        # Take the decoder output and output the desireed number of classes
        output = self.head(output)
        return output
