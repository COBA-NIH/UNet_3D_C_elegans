import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as 
from threedee.networks.threedee_unet import UNet3D_PixelShuffle

# class PixelShuffle3d(nn.Module):
#     def __init__(self, scale):
#         super().__init__()
#         self.scale = scale

#     def forward(self, input):
#         # 
#         batch_size, channels, in_depth, in_height, in_width = input.size()
#         nOut = channels // self.scale ** 3

#         out_depth = in_depth * self.scale
#         out_height = in_height * self.scale
#         out_width = in_width * self.scale

#         input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

#         output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

#         return output.view(batch_size, nOut, out_depth, out_height, out_width)

# use nn.PixelShuffle3d


class Discriminator(nn.Module):

    def __init__(
        self,
        patch_size,
        in_channels,
        **kwargs
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_size = tuple([int(p/2**4) for p in patch_size])

        # Define layer instances
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.leaky1 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm3d(128)
        self.leaky2 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm3d(256)
        self.leaky3 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm3d(512)
        self.leaky4 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv5 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.InstanceNorm3d(512)
        self.leaky5 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv6 = nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=1)
        self.sig6 = nn.Sigmoid()

    def forward(self, img):
        
        out = self.conv1(img)
        out = self.leaky1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.leaky2(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.leaky3(out)
        
        out = self.conv4(out)
        out = self.norm4(out)
        out = self.leaky4(out)
        
        out = self.conv5(out)
        out = self.norm5(out)
        out = self.leaky5(out)
        
        out = self.conv6(out)
        out = self.sig6(out)

        return out

class GAN(pl.LightningModule):
    def __init__(
        self,
        patch_size,
        in_channels,
        out_channels,
        feature_map,
        output_activation,
        normailisation_method
        ):
        super(GAN, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_map = feature_map
        self.output_activation = output_activation
        self.normailisation_method = normailisation_method

        self.generator = UNet3D_PixelShuffle(
            self.patch_size,
            self.in_channels,
            self.out_channels,
            self.feature_map,
            self.output_activation,
            self.normailisation_method
        )
        self.discriminator = Discriminator(
            self.patch_size,
            self.in_channels,
            self.out_channels,   
        )

        gen_state = {
            "prev_discrim_output": list(),
            
        }