import math
import torch
from torch import nn
import torch.nn.functional as F
from cf_frame.util import cheby
from cf_frame.configurator import args
from cf_frame.model.pf_chebii import PF_ChebII
uniformInit = nn.init.uniform

class PF_ChebII_Prefix(PF_ChebII): # Support: [-1,1]
    def __init__(self, data_handler):
        super().__init__(data_handler)

    def init_weight(self):
        if args.weights is not None:
            params = torch.tensor(args.weights)
        else:
            # params = uniformInit(torch.empty(args.order))
            params = torch.ones(args.order) * (2/args.order)
        self.initial = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.params = nn.Parameter(params)

    def get_coeffs(self):
        params = F.relu(self.params) # non-increasing constraint
        gammas = torch.zeros(args.order+1)
        gammas[0] = self.initial
        for i in range(1, args.order+1):
            gammas[i] = gammas[i-1] - params[i-1]
        gammas = F.relu(gammas) # non-negativity constraint
        coeffs = []
        K = args.order
        for k in range(K+1):
            coeff = torch.tensor(0.0).to(args.device)
            for j in range(K+1):
                x_j = math.cos((K-j+0.5) * math.pi / (K+1)) # Chebyshev node
                coeff += gammas[j] * cheby(k,x_j)
            coeff *= (2/(K+1)) 
            coeffs.append(coeff)
        coeffs[0] /= 2 # first term halved
        return torch.stack(coeffs)