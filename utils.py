import torch
import torch.nn as nn
import torch.nn.functional as F
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, input,target):
        target=F.interpolate(target,scale_factor=0.5)
        N = target.size(0)
        smooth = 1
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        return loss

class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.loss=nn.CrossEntropyLoss()
    def forward(self, input, target):
        target=F.interpolate(target,scale_factor=0.5)
        input = input.permute(0,2,3,1)
        target= target.permute(0,2,3,1)
        input=input.view(-1,59)
        target=target.view(-1)
        target=torch.Tensor.long(target)
        loss=self.loss(input,target)
        return loss
