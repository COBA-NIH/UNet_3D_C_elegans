import torch.nn as nn
import torch

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

        intersection = (prediction * target).sum()

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
