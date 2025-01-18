import torch
from torch import nn
from cf_frame.model import BaseModel
from cf_frame.configurator import args

init = nn.init.xavier_uniform_


class MF(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.user_embeds = nn.Parameter(init(torch.empty(self.user_num, args.embed_dim)))
        self.item_embeds = nn.Parameter(init(torch.empty(self.item_num, args.embed_dim)))

    def forward(self):
        return self.user_embeds, self.item_embeds

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
    