import torch.nn as nn
import torch
import numpy as np
from unet.utils.metrics import dice_coef

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
    
    def forward(self, prediction, target):
        prediction = nn.Sigmoid()(prediction)

        dice = dice_coef(prediction, target)
        print(f"dice_coef: {dice}")
        return 1. - dice

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

class WeightedBCELoss(nn.Module):
    def __init__(self, class_weights=None, per_image=False, per_channel=False):
        super().__init__()
        self.class_weights = class_weights
        self.per_image = per_image
        self.per_channel = per_channel
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, prediction, target, weight_map=None):
        prediction = prediction.sigmoid().flatten()
        target = target.flatten().float()
        weight_map = weight_map.flatten().float()

        loss = self.bce(prediction, target) * weight_map

        loss = loss.mean()

        return loss

class WeightedDiceLoss(nn.Module):
    """Pixel weighted dice loss"""
    def __init__(self, class_weights=None, per_image=False, per_channel=False):
        super().__init__()
        self.class_weights = class_weights
        self.per_image = per_image
        self.per_channel = per_channel
    
    def forward(self, prediction, target, weight_map):
        prediction = prediction.sigmoid().flatten()
        target = target.flatten().float()
        weight_map = weight_map.flatten().float()

        intersection = (prediction * target * weight_map).sum()
        dice_coef = 2 * (intersection / ((prediction ** 2).sum() + (target ** 2).sum()).clamp(min=1e-6))
        return 1. - dice_coef

class WeightedBCEDiceLoss(nn.Module):
    """Pixel weighted BCE + Dice Loss"""
    def __init__(self, class_weights=None, per_image=False, per_channel=False):
        super().__init__()
        self.class_weights = class_weights
        self.bce = WeightedBCELoss(class_weights=None, per_image=per_image, per_channel=per_channel)
        self.dice = WeightedDiceLoss(class_weights=None, per_image=per_image, per_channel=per_channel)

    def forward(self, prediction, target, weight_map=None):
        loss = self.bce(prediction, target, weight_map) + self.dice(prediction, target, weight_map)
        return loss