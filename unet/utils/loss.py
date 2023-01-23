import torch.nn as nn
import torch
import numpy as np

# def dice_loss(input, target, epsilon=1e-6):
#     assert input.shape == target.shape, "Input must be same shape as target"

#     input = flatten(input)
#     target = flatten(target)
#     target = target.float()

#     intersection = (input * target).sum(-1)
    
#     denominator = (input * input).sum(-1) + (target * target).sum(-1)
#     # You can clamp the output within a certain min-max, here epsilon  
#     return 2 * (intersect / denominator.clamp(min=epsilon))



class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def dice_loss(self, prediction, target):
        # Make arrays 1D
        prediction = prediction.flatten()
        target = target.flatten()
        target = target.float()

        intersection = (prediction * target).sum(-1)
        print("intersection", intersection, prediction.shape, target.shape)
        intersection = torch.tensor[1, 3, 3] * intersection

        loss = 1 - 2 * (intersection) / (prediction.sum() + target.sum())
        return loss
    
    def forward(self, prediction, target):
        prediction = nn.Sigmoid()(prediction)

        dice = self.dice_loss(prediction, target)

        avg_channel_dice = 1. - torch.mean(dice)

        return avg_channel_dice
        # return dice

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, prediction, target):
        output = self.bce(prediction, target.float()) + self.dice(prediction, target)

        return output


class WeightedBCELoss(nn.Module):
    def __init__(self, per_image=False, per_channel=False):
        super().__init__()
        self.per_image = per_image
        self.per_channel = per_channel
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, prediction, target, weights):
        batch_size, n_channels = prediction.size(0), prediction.size(1)

        # n_parts to split tensor, dependeing on per-channel/image
        n_parts = 1
        if self.per_image:
            n_parts = batch_size
        if self.per_channel:
            n_parts = batch_size * n_channels

        # -1 to infer the shape based on n_parts
        # prediction = prediction.contiguous().view(n_parts, -1)
        # target = target.contiguous().view(n_parts, -1)
        # weights = weights.contiguous().view(n_parts, -1)
        prediction = prediction.view(n_parts, -1)
        target = target.view(n_parts, -1)
        weights = weights.view(n_parts, -1)

        loss = self.bce(prediction, target) * weights

        loss = loss.mean()

        return loss

class WeightedDiceLoss(nn.Module):
    """Pixel weighted dice loss"""
    def __init__(self, per_image=False, per_channel=False):
        self.per_image = per_image
        self.per_channel = per_channel
        super().__init__()
    
    def forward(self, prediction, target, weight_map):
        eps = 1e-6
        # n_parts to split tensor, dependeing on per-channel/image
        n_parts = 1
        if self.per_image:
            n_parts = batch_size
        if self.per_channel:
            n_parts = batch_size * n_channels

        # -1 to infer the shape based on n_parts
        # prediction = prediction.contiguous().view(n_parts, -1)
        # target = target.contiguous().view(n_parts, -1)
        # weights = weights.contiguous().view(n_parts, -1)
        prediction = prediction.view(n_parts, -1)
        target = target.view(n_parts, -1)
        weights = weights.view(n_parts, -1)

        intersection = torch.sum(dice_output * dice_target * weights, dim=1)
        union = torch.sum(prediction, dim=1) + torch.sum(target, dim=1) + eps
        loss = (1 - (2 * intersection + eps) / union).mean()
        return loss



### ComboLoss - various sources

def soft_dice_loss(outputs, targets, per_image=False, per_channel=False):
    batch_size, n_channels = outputs.size(0), outputs.size(1)
    
    eps = 1e-6
    n_parts = 1
    if per_image:
        n_parts = batch_size
    if per_channel:
        n_parts = batch_size * n_channels
    
    dice_target = targets.contiguous().view(n_parts, -1).float()
    dice_output = outputs.contiguous().view(n_parts, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss

# def dice_metric(preds, trues, per_image=False, per_channel=False):
#     preds = preds.float()
#     return 1 - soft_dice_loss(preds, trues, per_image, per_channel)


# def jaccard(outputs, targets, per_image=False, non_empty=False, min_pixels=5):
#     batch_size = outputs.size()[0]
#     eps = 1e-3
#     if not per_image:
#         batch_size = 1
#     dice_target = targets.contiguous().view(batch_size, -1).float()
#     dice_output = outputs.contiguous().view(batch_size, -1)
#     target_sum = torch.sum(dice_target, dim=1)
#     intersection = torch.sum(dice_output * dice_target, dim=1)
#     losses = 1 - (intersection + eps) / (torch.sum(dice_output + dice_target, dim=1) - intersection + eps)
#     if non_empty:
#         assert per_image == True
#         non_empty_images = 0
#         sum_loss = 0
#         for i in range(batch_size):
#             if target_sum[i] > min_pixels:
#                 sum_loss += losses[i]
#                 non_empty_images += 1
#         if non_empty_images == 0:
#             return 0
#         else:
#             return sum_loss / non_empty_images

#     return losses.mean()


# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True, per_image=False):
#         super().__init__()
#         self.size_average = size_average
#         self.register_buffer('weight', weight)
#         self.per_image = per_image

#     def forward(self, input, target):
#         return soft_dice_loss(input, target, per_image=self.per_image)


# class JaccardLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True, per_image=False, non_empty=False, apply_sigmoid=False,
#                  min_pixels=5):
#         super().__init__()
#         self.size_average = size_average
#         self.register_buffer('weight', weight)
#         self.per_image = per_image
#         self.non_empty = non_empty
#         self.apply_sigmoid = apply_sigmoid
#         self.min_pixels = min_pixels

#     def forward(self, input, target):
#         if self.apply_sigmoid:
#             input = torch.sigmoid(input)
#         return jaccard(input, target, per_image=self.per_image, non_empty=self.non_empty, min_pixels=self.min_pixels)

# class FocalLoss2d(nn.Module):
#     def __init__(self, gamma=2, ignore_index=255):
#         super().__init__()
#         self.gamma = gamma
#         self.ignore_index = ignore_index

#     def forward(self, outputs, targets):
#         outputs = outputs.contiguous()
#         targets = targets.contiguous()
#         eps = 1e-8
#         non_ignored = targets.view(-1) != self.ignore_index
#         targets = targets.view(-1)[non_ignored].float()
#         outputs = outputs.contiguous().view(-1)[non_ignored]
#         outputs = torch.clamp(outputs, eps, 1. - eps)
#         targets = torch.clamp(targets, eps, 1. - eps)
#         pt = (1 - targets) * (1 - outputs) + targets * outputs
#         return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()


# class ComboLoss(nn.Module):
#     def __init__(self, weights, per_image=False, channel_weights=[1, 0.5, 0.5], channel_losses=None):
#         super().__init__()
#         self.weights = weights
#         self.bce = nn.BCEWithLogitsLoss()
#         self.dice = DiceLoss(per_image=False)
#         self.jaccard = JaccardLoss(per_image=False)
#         # self.lovasz = LovaszLoss(per_image=per_image)
#         # self.lovasz_sigmoid = LovaszLossSigmoid(per_image=per_image)
#         self.focal = FocalLoss2d()
#         self.mapping = {'bce': self.bce,
#                         'dice': self.dice,
#                         'focal': self.focal,
#                         'jaccard': self.jaccard,
#                         # 'lovasz': self.lovasz,
#                         # 'lovasz_sigmoid': self.lovasz_sigmoid
#                         }
#         self.expect_sigmoid = {'dice', 'focal', 'jaccard', 'lovasz_sigmoid'}
#         self.per_channel = {'dice', 'jaccard', 'lovasz_sigmoid'}
#         self.values = {}
#         self.channel_weights = channel_weights
#         self.channel_losses = channel_losses

#     def forward(self, outputs, targets):
#         loss = 0
#         weights = self.weights
#         sigmoid_input = torch.sigmoid(outputs)
#         for k, v in weights.items():
#             if not v:
#                 continue
#             val = 0
#             if k in self.per_channel:
#                 channels = targets.size(1)
#                 for c in range(channels):
#                     if not self.channel_losses or k in self.channel_losses[c]:
#                         val += self.channel_weights[c] * self.mapping[k](sigmoid_input[:, c, ...] if k in self.expect_sigmoid else outputs[:, c, ...],
#                                                targets[:, c, ...])

#             else:
#                 val = self.mapping[k](sigmoid_input if k in self.expect_sigmoid else outputs, targets)

#             self.values[k] = val
#             loss += self.weights[k] * val
#         return loss.clamp(min=1e-5)