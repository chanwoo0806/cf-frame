import torch
from torch import nn
from cf_frame.model import BaseModel
from cf_frame.module import EdgeDrop
from cf_frame.configurator import args

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class LightGCN(BaseModel):
    def __init__(self, data_handler):
        super().__init__()

        self.adj = data_handler.torch_adj

        self.embed_dim = args.embed_dim
        self.layer_num = args.layer_num
        self.keep_rate = args.keep_rate

        self.user_embeds = nn.Parameter(init(torch.empty(self.user_num, self.embed_dim)))
        self.item_embeds = nn.Parameter(init(torch.empty(self.item_num, self.embed_dim)))

        self.edge_dropper = EdgeDrop()
        self.final_embeds = None

    def _propagate(self, adj, embeds):
        return torch.spmm(adj, embeds)

    def forward(self, adj, keep_rate):
        if not self.training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        embeds = torch.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        if self.training:
            adj = self.edge_dropper(adj, keep_rate)
        for _ in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = torch.stack(embeds_list, dim=0).mean(dim=0)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:]

    def full_predict(self, batch_data):
        if self.training:
            raise Exception('full_predict should be called in eval mode')
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
    