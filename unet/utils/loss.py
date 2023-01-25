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

class AbstractWeightedLoss(nn.Module):
    """Class for abstracting weighted loss methods. Allows for unreduced tensors
    to be multiplied by weight maps or just indiscriminate class weights"""
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights

    def resolve_weights(self, target, weight_map=None):
        batch_size, n_channels = target.size(0), target.size(1)
        if self.class_weights is not None and weight_map is None:
            assert len(self.class_weights) == n_channels, f"class_weight {self.class_weights} does not match number of target classes {n_channels}"
            weights = torch.ones((target.size()))
            for c in range(n_channels):
                weights[:,c,...] = weights[:,c,...] * self.class_weights[c]
            return weights
        elif self.class_weights is None and weight_map is not None:
            return weight_map
        elif self.class_weights is None and weight_map is None:
            return torch.ones((target.size()))
        elif self.class_weights is not None and weight_map is not None:
            raise ValueError("Both class_weights and weight_map are requested but you can only have one")
        else:
            raise NotImplementedError
class WeightedBCELoss(AbstractWeightedLoss):
    def __init__(self, class_weights=None, per_image=False, per_channel=False):
        super().__init__(class_weights)
        self.class_weights = class_weights
        self.per_image = per_image
        self.per_channel = per_channel
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, prediction, target, weights=None):
        batch_size, n_channels = prediction.size(0), prediction.size(1)

        # n_parts to split tensor, dependeing on per-channel/image
        # I don't think this is necessary...
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

class WeightedDiceLoss(AbstractWeightedLoss):
    """Pixel weighted dice loss"""
    def __init__(self, device, class_weights=None, per_image=False, per_channel=False):
        super().__init__(device, class_weights)
        self.class_weights = class_weights
        self.per_image = per_image
        self.per_channel = per_channel
    
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
        prediction = prediction.sigmoid().view(n_parts, -1)
        target = target.view(n_parts, -1)
        weights = weights.view(n_parts, -1)

        intersection = torch.sum(dice_output * dice_target * weights, dim=1)
        union = torch.sum(prediction, dim=1) + torch.sum(target, dim=1) + eps
        loss = (1 - (2 * intersection) / union).mean()
        return loss


class WeightedBCEDiceLoss(AbstractWeightedLoss):
    """Pixel weighted BCE + Dice Loss"""
    def __init__(self, class_weights=None, per_image=False, per_channel=False):
        super().__init__(class_weights)
        self.class_weights = class_weights
        self.bce = WeightedBCELoss(class_weights=None, per_image=per_image, per_channel=per_channel)
        self.dice = WeightedBCELoss(class_weights=None, per_image=per_image, per_channel=per_channel)

    def forward(self, prediction, target, weight_map=None):
        weights = self.resolve_weights(target, weight_map).to(target.device)
        loss = self.bce(prediction, target, weights) + self.dice(prediction, target, weights)
        return loss


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

