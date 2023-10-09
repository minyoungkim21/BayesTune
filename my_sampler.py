import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Sampler(nn.Module):
    def __init__(self, theta, alp, bet, lamb_init, Nexp, lr):
        super().__init__()

        names, shapes, nweights = [], [], 0
        for pn, p in theta.named_parameters():
            names.append(pn)
            shapes.append(list(p.shape))
            nweights += p.numel()

        # lambda
        # self.lamb_ = nn.ParameterList([
        #     nn.Parameter( np.log(lamb_init) * torch.ones(*shape) ) for shape in shapes
        # ])
        self.lamb_ = nn.ParameterList([
            nn.Parameter( lamb_init * torch.ones(*shape) ) for shape in shapes
        ])

        self.layer_names = names
        self.nweights = nweights
        self.N = 10**Nexp
        self.lr = lr
        self.alp = alp
        self.bet = bet

    def _transform_positive(self, params_list):         
        #return [torch.exp(p) for p in params_list]
        return [torch.clamp(p,min=1e-8) for p in params_list]

    def create_optimizer(self):
        self.optimizer = torch.optim.Adam(
            [{'params': list(self.lamb_.parameters()), 'lr': self.lr},], 
        )

    def print_stats(self):
        ret_str = ''
        lamb_layers = self._transform_positive(self.lamb_)
        internal = nn.utils.parameters_to_vector(lamb_layers)
        internal_str = '(avg) %.6f, (min) %.6f, (max) %.6f\n' % (internal.mean().item(), internal.min().item(), internal.max().item(), )
        qp = np.quantile(internal.to('cpu').detach().numpy(), [0.9, 0.99, 0.999])
        internal_str += '            (median) %.6f, (0.9-tile) %.6f, (0.99-tile) %.6f, (0.999-tile) %.6f\n' % (
            internal.median().item(), qp[0], qp[1], qp[2])
        ret_str += '    lambda: %s\n' % (internal_str,)
        return ret_str

    def print_lambda_sparsity(self, theta, bs_mask):
        ret_str = ''
        for pn, p in theta.named_parameters():
            lidx = self.layer_names.index(pn)
            ret_str += '    %s: %.6f%% -- can be updated\n' % (
                pn, 100.*(bs_mask[lidx]==False).sum().item()/bs_mask[lidx].numel())
        return ret_str


    def print_sparsity_patterns(self, theta, bs_mask):
        strout = []
        for pn, p in theta.named_parameters():
            lidx = self.layer_names.index(pn)
            if 'embeddings' in pn:
                layer_num = -1
            elif 'taskmodels' in pn or 'pooler' in pn:
                layer_num = 12
            else:
                layer_num = int(pn.split('.')[3])
            strout.append([pn,layer_num, (bs_mask[lidx]==False).sum().item(), bs_mask[lidx].numel(),self.nweights])
        ret_str = '\n====\n' + str(strout) + '\n====\n'

        return ret_str    
