import math
import torch
from torch import nn
import torch.nn.functional as F
from cf_frame.util import cheby
from cf_frame.configurator import args
from cf_frame.model.pf_cheb import PF_Cheb
uniformInit = nn.init.uniform

class PF_ChebII(PF_Cheb): # Support: [-1,1]
    def __init__(self, data_handler):
        super().__init__(data_handler)
        
    def init_weight(self):
        if args.weights is not None:
            params = torch.tensor(args.weights)
        else:
            # params = uniformInit(torch.empty(args.order+1))
            params = torch.zeros(args.order+1)
            params[0] = 2
            for i in range(1, args.order+1):
                params[i] = params[i-1] - (2/args.order)
        self.params = nn.Parameter(params)

    def get_coeffs(self):
        params = F.relu(self.params) # non-negativity constraint
        coeffs = []
        K = args.order
        for k in range(K+1):
            coeff = torch.tensor(0.0).to(args.device)
            for j in range(K+1):
                x_j = math.cos((K-j+0.5) * math.pi / (K+1)) # Chebyshev node
                coeff += params[j] * cheby(k,x_j)
            coeff *= (2/(K+1)) 
            coeffs.append(coeff)
        coeffs[0] /= 2 # the first term is to be halved
        return torch.stack(coeffs)