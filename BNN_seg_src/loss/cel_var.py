import torch
import torch.nn as nn
import torch.nn.functional as F


class CEL_VAR(nn.Module):
    def __init__(self, var_weight):
        super(CEL_VAR, self).__init__()
        self.var_weight = var_weight

    def forward(self, results, label):
        mean, log_var = results['mean'], results['var']
        log_var = self.var_weight * log_var
        var = torch.exp(log_var)
        std = torch.sqrt(var)
        undistort_loss =  F.cross_entropy(mean, label)
        distorted_loss = self.gaussian_categorical_cross_entropy(mean,std,label,num_mc_sample=10)
        variance_regularize = var.mean()

        return distorted_loss # + variance_regularize # + undistort_loss

    def gaussian_categorical_cross_entropy(self,mean,std,label,num_mc_sample=10):
        """
        batch x classes x height x width
        each pixel have one mean, std, sample num_mc_sample times
        """
        distorted_loss_list = torch.zeros(num_mc_sample)

        for sample_idx in range(num_mc_sample):
            eps = torch.rand(size = std.size()).cuda() # # ([128, 5, 96, 64])
            logits_sample = mean+std*eps
            distorted_loss_list[sample_idx]  = F.nll_loss(F.log_softmax(logits_sample, dim=1), label, reduction='mean')
            #F.cross_entropy(logits_sample, label)
            

        return distorted_loss_list.mean()


    

