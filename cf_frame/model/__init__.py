import torch
from torch import nn
from cf_frame.configurator import args

class BaseModel(nn.Module):
    def __init__(self, data_handler):
        super().__init__()
        # put data_handler.xx you need into self.xx
        # put hyperparams you need into self.xx
        # initialize parameters
        self.user_num = data_handler.user_num
        self.item_num = data_handler.item_num
    
    def forward(self):
        """return final embeddings for all users and items
        Return:
            user_embeds (torch.Tensor): user embeddings
            item_embeds (torch.Tensor): item embeddings
        """
        pass
    
    def full_predict(self, batch_data):
        """return all-rank predictions to evaluation process, should call _mask_predict for masking the training pairs
        Args:
            batch_data (tuple): data in a test batch, e.g. batch_users, train_mask
        Return:
            full_preds (torch.Tensor): a [test_batch_size * item_num] prediction tensor
        """
        pass

    def _mask_predict(self, full_preds, train_mask):
        return full_preds * (1 - train_mask) - 1e8 * train_mask
        