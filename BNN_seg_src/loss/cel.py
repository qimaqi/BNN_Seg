import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice_score import dice_loss
class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()

    def forward(self, results, label):
        mean = results['mean'] # [128, 3, 96, 64]
        loss = F.cross_entropy(mean, label) \
            # + dice_loss(F.softmax(mean, dim=1).float(),
            #                            F.one_hot(label, 5).permute(0, 3, 1, 2).float(),
            #                            multiclass=True)
            
            # F.nll_loss(F.log_softmax(mean, dim = ?),label.reduction='mean')
        return loss
