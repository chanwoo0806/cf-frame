import torch
from torch import nn
from cf_frame.model import BaseModel
from cf_frame.module import EdgeDrop
from cf_frame.configurator import args
from cf_frame.util import scipy_coo_to_torch_sparse
# JGCF
from functools import partial

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


def JacobiConv(L, xs, adj, alphas, a=1.0, b=1.0, l=-1.0, r=1.0):
    '''
    Jacobi Bases. Please refer to our paper for the form of the bases.
    '''
    if L == 0: return xs[0]
    if L == 1:
        coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)
        coef1 *= alphas[0]
        coef2 = (a + b + 2) / (r - l)
        coef2 *= alphas[0]
        return coef1 * xs[-1] + coef2 * (adj @ xs[-1])
    coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
    coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
    coef_lm1_2 = (2 * L + a + b - 1) * (a**2 - b**2)
    coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)
    tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)
    tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)
    tmp3 = alphas[L - 1] * alphas[L - 2] * (coef_lm2 / coef_l)
    tmp1_2 = tmp1 * (2 / (r - l))
    tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
    nx = tmp1_2 * (adj @ xs[-1]) - tmp2_2 * xs[-1]
    nx -= tmp3 * xs[-2]
    return nx


class PolyConvFrame(nn.Module):
    def __init__(self, conv_fn, depth=3, alpha=1.0, fixed=True):
        super().__init__()
        self.depth = depth
        self.basealpha = alpha
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(min(1 / alpha, 1))), requires_grad=not fixed) for i in range(depth + 1)
        ])
        self.conv_fn = conv_fn

    def forward(self, x, adj):
        '''
        Args:
            x: node embeddings. of shape (number of nodes, node feature dimension)
        '''
        alphas = [self.basealpha * torch.tanh(_) for _ in self.alphas]
        xs = [self.conv_fn(0, [x], adj, alphas)]
        for L in range(1, self.depth + 1):
            tx = self.conv_fn(L, xs, adj, alphas)
            xs.append(tx)
        xs = [x.unsqueeze(1) for x in xs]
        x = torch.cat(xs, dim=1)
        return x


class JGCF(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)

        adj = data_handler.get_normalized_adj()
        self.adj = scipy_coo_to_torch_sparse(adj)

        self.embed_dim = args.embed_dim
        self.layer_num = args.layer_num
        self.keep_rate = args.keep_rate

        self.user_embeds = nn.Parameter(init(torch.empty(self.user_num, self.embed_dim)))
        self.item_embeds = nn.Parameter(init(torch.empty(self.item_num, self.embed_dim)))

        self.edge_dropper = EdgeDrop()
        self.final_embeds = None
        self.device = args.device

        # JGCF
        self.a = args.a
        self.b = args.b
        self.beta = args.alpha

        conv_fn = partial(JacobiConv, a=self.a, b=self.b)
        self.graph_conv_low = PolyConvFrame(
            conv_fn=conv_fn, depth=self.layer_num, alpha=3.0
        )

    def forward(self):
        if not self.training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        embeds = torch.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_low = self.graph_conv_low(embeds, self.adj)
        embeds_low = embeds_low.mean(1)
        embeds_mid = self.beta * embeds - embeds_low
        embeds = torch.hstack([embeds_low, embeds_mid])
        self.final_embeds = embeds
        user_embeddings, item_embeddings = torch.split(embeds, [self.user_num, self.item_num])
        return user_embeddings, item_embeddings

    def full_predict(self, batch_data):
        if self.training:
            raise Exception('full_predict should be called in eval mode')
        user_embeds, item_embeds = self.forward()
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
    