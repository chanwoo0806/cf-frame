import torch
from torch import nn
from cf_frame.model import BaseModel
from cf_frame.module import EdgeDrop
from cf_frame.configurator import args

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


# It is equal to MF.
class UltraGCN(BaseModel):
    def __init__(self):
        self.embed_dim = args.embed_dim
        self.user_embeds = nn.Parameter(init(torch.empty(self.user_num, self.embed_dim)))
        self.item_embeds = nn.Parameter(init(torch.empty(self.item_num, self.embed_dim)))

    def forward(self):
        return self.user_embeds, self.item_embeds

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward()
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
    