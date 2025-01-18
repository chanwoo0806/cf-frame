import torch
from torch import nn
import torch.nn.functional as F

from cf_frame.model import BaseModel
from cf_frame.module import EdgeDrop
from cf_frame.configurator import args
from cf_frame.util import scipy_coo_to_torch_sparse

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class NGCF(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        
        adj = data_handler.get_normalized_adj()
        self.adj = scipy_coo_to_torch_sparse(adj)

        self.embed_dim = args.embed_dim
        self.weight_size = args.weight_size
        self.layer_num = len(self.weight_size)
        self.dropout_rates = args.dropout_rates
        self.keep_rate = args.keep_rate

        self.user_embeds = nn.Parameter(init(torch.empty(self.user_num, self.embed_dim)))
        self.item_embeds = nn.Parameter(init(torch.empty(self.item_num, self.embed_dim)))

        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()

        layer_sizes = [self.embed_dim] + self.weight_size
        for i in range(self.layer_num):
            self.GC_Linear_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.Bi_Linear_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.dropout_list.append(nn.Dropout(self.dropout_rates[i]))

        self.edge_dropper = EdgeDrop()
        self.final_embeds = None

    def _propagate(self, adj, ego_embeddings):
        side_embeddings = torch.spmm(adj, ego_embeddings)
        sum_embeddings = F.leaky_relu(self.GC_Linear_list[0](side_embeddings))
        bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
        bi_embeddings = F.leaky_relu(self.Bi_Linear_list[0](bi_embeddings))
        return sum_embeddings + bi_embeddings

    def forward(self):
        if not self.training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        ego_embeddings = torch.cat([self.user_embeds, self.item_embeds], dim=0)
        adj = self.edge_dropper(self.adj, self.keep_rate) if self.training else self.adj
        all_embeddings = [ego_embeddings]
        for i in range(self.layer_num):
            ego_embeddings = self._propagate(adj, ego_embeddings)
            ego_embeddings = self.dropout_list[i](ego_embeddings)
            ego_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(ego_embeddings)
        final_embeddings = torch.cat(all_embeddings, dim=1)
        self.final_embeds = final_embeddings
        return final_embeddings[:self.user_num], final_embeddings[self.user_num:]

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