import torch
from torch import nn, spmm
from cf_frame.configurator import args
from cf_frame.model import PolyFilter
uniformInit = nn.init.uniform

class PF_Mono(PolyFilter): # Support: [0,1]
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.type = 'mono'
        
    def init_weight(self):
        if args.weights is not None:
            params = torch.tensor(args.weights)
        else:
            # params = uniformInit(torch.empty(args.order+1))
            params = torch.zeros(args.order+1)
            params[0:2] = torch.tensor([1.0, -1.0])
        self.params = nn.Parameter(params)

    def L_tilde(self, norm_inter, x):
        # L_tilde = L/lambda_max = L/1 = L = I - R_tilde^T*R_tilde
        y = spmm(norm_inter, x)
        y = spmm(norm_inter.t(), y) * (-1)
        y += x
        return y

    def get_bases(self, signal, norm_inter):
        bases = []
        bases.append(signal) # x[0]
        for _ in range(1, args.order+1):
            # x[k] = L * x[k-1]
            basis = self.L_tilde(norm_inter, bases[-1])
            bases.append(basis)
        return torch.stack(bases, dim=0)

    def get_coeffs(self):
        return self.params