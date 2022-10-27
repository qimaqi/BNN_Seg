import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        print('Making model...')

        self.is_train = config.is_train
        self.num_gpu = config.num_gpu
        if config.uncertainty[-1] == 'c':
            self.uncertainty = config.uncertainty[:-2]
        else:
            self.uncertainty = config.uncertainty

        self.n_samples = config.n_samples
        module = import_module('model.' + self.uncertainty)
        self.model = module.make_model(config).to(config.device)

    def forward(self, input):
        if self.model.training:
            if self.num_gpu > 1:
                return P.data_parallel(self.model, input,
                                       list(range(self.num_gpu)))
            else:
                return self.model.forward(input)
        else:
            forward_func = self.model.forward
            if self.uncertainty == 'normal':
                return forward_func(input)
            if self.uncertainty == 'aleatoric':
                return self.test_aleatoric(input, forward_func)
            elif self.uncertainty == 'epistemic':
                return self.test_epistemic(input, forward_func)
            elif self.uncertainty == 'combined':
                return self.test_combined(input, forward_func)

    def test_aleatoric(self, input, forward_func):
    
        def calc_entropy(input_tensor):
            lsm = nn.LogSoftmax(dim=1)
            log_probs = lsm(input_tensor) #
            probs = torch.exp(log_probs)
            p_log_p = log_probs * probs
            entropy = -p_log_p.mean(dim=1)
            return entropy
        

        mean_samples = []
        var_a_samples = []
        # n_samples model samples
        for i_sample in range( self.n_samples):
            results = forward_func(input)
            mean_i = results['mean']
            var_i = results['var']
            var_i = torch.exp(var_i)
            mean_samples.append(mean_i)
            var_a_samples.append(var_i)



        mean = torch.stack(mean_samples, dim=0).mean(dim=0)

        a_var = torch.stack(var_a_samples, dim=0).mean(dim=0) #  B X C X H x W
        a_std = torch.sqrt(a_var)
        a_var_entropy_high = calc_entropy(mean+a_std)
        a_var_entropy_low = calc_entropy(mean-a_std)
        a_var = torch.abs(a_var_entropy_high - a_var_entropy_low)

        a_var = a_var/a_var.max()


        results = {'mean': mean, 'a_var':a_var }
        return results

    def test_epistemic(self, input, forward_func):
        def calc_entropy(input_tensor):
            lsm = nn.LogSoftmax(dim=1)
            log_probs = lsm(input_tensor) #
            probs = torch.exp(log_probs)
            p_log_p = log_probs * probs
            entropy = -p_log_p.mean(dim=1)
            return entropy
        
        mean_samples = []
        # n_samples model samples
        for i_sample in range( self.n_samples):
            results = forward_func(input)
            mean_i = results['mean']
            mean_samples.append(mean_i)

        mean = torch.stack(mean_samples, dim=0).mean(dim=0)

        e_var =  torch.stack(mean_samples, dim=0).var(dim=0) # 
        e_var = torch.sqrt(e_var)
        e_var_entropy_high = calc_entropy(mean+e_var)
        e_var_entropy_low = calc_entropy(mean-e_var)
        e_variance = torch.abs(e_var_entropy_high - e_var_entropy_low)

        e_variance = e_variance/e_variance.max()


        results = {'mean': mean, 'e_var': e_variance}
        return results

    def test_combined(self, input, forward_func):
    
        def calc_entropy(input_tensor):
            lsm = nn.LogSoftmax(dim=1)
            log_probs = lsm(input_tensor) #
            probs = torch.exp(log_probs)
            p_log_p = log_probs * probs
            entropy = -p_log_p.mean(dim=1)
            return entropy
        

        mean_samples = []
        var_e_samples = []
        var_a_samples = []
        # n_samples model samples
        for i_sample in range( self.n_samples):
            results = forward_func(input)
            mean_i = results['mean']
            mean_entropy_i = calc_entropy(mean_i)
            var_i = results['var']
            var_i = torch.exp(var_i)
            mean_samples.append(mean_i)
            var_a_samples.append(var_i)
            var_e_samples.append(mean_entropy_i)


        mean = torch.stack(mean_samples, dim=0).mean(dim=0)

        a_var = torch.stack(var_a_samples, dim=0).mean(dim=0) #  B X C X H x W
        a_std = torch.sqrt(a_var)
        a_var_entropy_high = calc_entropy(mean+a_std)
        a_var_entropy_low = calc_entropy(mean-a_std)
        a_var = torch.abs(a_var_entropy_high - a_var_entropy_low)

        a_var = a_var/a_var.max()

        e_var =  torch.stack(mean_samples, dim=0).var(dim=0) # 
        e_var = torch.sqrt(e_var)
        e_var_entropy_high = calc_entropy(mean+e_var)
        e_var_entropy_low = calc_entropy(mean-e_var)
        e_variance = torch.abs(e_var_entropy_high - e_var_entropy_low)

        e_variance = e_variance/e_variance.max()


        results = {'mean': mean, 'e_var': e_variance, 'a_var':a_var }
        return results

    def save(self, ckpt, epoch):
        save_dirs = [os.path.join(ckpt.model_dir, 'model_latest.pt')]
        save_dirs.append(
            os.path.join(ckpt.model_dir, 'model_{}.pt'.format(epoch)))
        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, ckpt, cpu=False):
        epoch = ckpt.last_epoch
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        if epoch == -1:
            load_from = torch.load(
                os.path.join(ckpt.model_dir, 'model_latest.pt'), **kwargs)
        else:
            load_from = torch.load(
                os.path.join(ckpt.model_dir, 'model_{}.pt'.format(epoch)), **kwargs)
        if load_from:
            self.model.load_state_dict(load_from, strict=False)
