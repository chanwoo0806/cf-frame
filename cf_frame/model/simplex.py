import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from cf_frame.model import BaseModel
from cf_frame.module import EdgeDrop
from cf_frame.configurator import args


init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SimpleX(BaseModel):
    def __init__(self, data_handler):
        super(SimpleX, self).__init__()

        # Aggregation
        self.aggregator        = args.aggregator
        self.fusing_weight     = args.fusing_weight
        self.attention_dropout = args.attention_dropout
        self.history_num       = args.history_num

        # Model
        self.embed_dim    = args.embed_dim
        self.score        = args.score
        self.dropout_rate = args.dropout

        # Interacted Items
        list_interacted = data_handler.train_dataloader.dataset.interacted_items
        processed = []
        for items in list_interacted:
            n_items = len(items)
            if n_items > self.history_num:
                items = random.sample(items, self.history_num)  # Chunking
            else:
                items += [-1] * (self.history_num - n_items)  # Padding
            processed.append(items)
        self.interacted = np.array(processed)

        self.behavior_aggregation = BehaviorAggregator(
            embedding_dim=self.embed_dim, 
            gamma=self.fusing_weight,
            aggregator=self.aggregator, 
            dropout_rate=self.attention_dropout
        )

        self.user_embeds = nn.Parameter(init(torch.empty(self.user_num, self.embed_dim)))
        self.item_embeds = nn.Parameter(init(torch.empty(self.item_num, self.embed_dim)))
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self):
        # user_interacted = b x seq_len x embedding_dim
        interacted_embeds = self.item_embeds[self.interacted]
        mask = (self.interacted == -1)
        interacted_embeds[mask] = 0
        aggregated_user_embeds = self.behavior_aggregation(self.user_embeds, interacted_embeds)

        if self.training:
            aggregated_user_embeds = self.dropout(aggregated_user_embeds)

        return aggregated_user_embeds, self.item_embeds
        

    def full_predict(self, batch_data):
        if self.training:
            raise Exception('full_predict should be called in eval mode')
        user_embeds, item_embeds = self.forward()
        if self.score == 'cosine':
            user_embeds = F.normalize(user_embeds)
            item_embeds = F.normalize(item_embeds)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds


# Copied from origin
class BehaviorAggregator(nn.Module):
    def __init__(self, embedding_dim, gamma=0.5, aggregator="mean", dropout_rate=0.):
        super(BehaviorAggregator, self).__init__()
        self.aggregator = aggregator
        self.gamma = gamma
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if self.aggregator in ["user_attention", "self_attention"]:
            self.W_k = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                     nn.Tanh())
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
            if self.aggregator == "self_attention":
                self.W_q = nn.Parameter(torch.Tensor(embedding_dim, 1))
                nn.init.xavier_normal_(self.W_q)

    def forward(self, uid_emb, sequence_emb):
        out = uid_emb
        if self.aggregator == "mean":
            out = self.average_pooling(sequence_emb)
        elif self.aggregator == "user_attention":
            out = self.user_attention(uid_emb, sequence_emb)
        elif self.aggregator == "self_attention":
            out = self.self_attention(sequence_emb)
        return self.gamma * uid_emb + (1 - self.gamma) * out

    def user_attention(self, uid_emb, sequence_emb):
        key = self.W_k(sequence_emb) # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.bmm(key, uid_emb.unsqueeze(-1)).squeeze(-1) # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def self_attention(self, sequence_emb):
        key = self.W_k(sequence_emb) # b x seq_len x attention_dim
        mask = sequence_emb.sum(dim=-1) == 0
        attention = torch.matmul(key, self.W_q).squeeze(-1) # b x seq_len
        attention = self.masked_softmax(attention, mask)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), sequence_emb).squeeze(1)
        return self.W_v(output)

    def average_pooling(self, sequence_emb):
        mask = sequence_emb.sum(dim=-1) != 0
        mean = sequence_emb.sum(dim=1) / (mask.float().sum(dim=-1, keepdim=True) + 1.e-9)
        return self.W_v(mean)

    def masked_softmax(self, X, mask):
        # use the following softmax to avoid nans when a sequence is entirely masked
        X = X.masked_fill_(mask, 0)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1.e-9)