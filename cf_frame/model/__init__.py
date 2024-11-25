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


class PolyFilter(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.init_weight()
        self.data_handler = data_handler
        self.data_handler.pf_set_inter()
        if args.ideal_num:
            self.data_handler.pf_set_ideal()
            
    def init_weight(self):
        pass
        
    def get_bases(self, signal, norm_inter):
        '''return bases for the polynomial
        Args:
            signal (torch.dense): signal matrix (N,B)
            norm_inter (torch.sparse): normalized interaction matrix (N,N)
        Return:
            bases (torch.dense): bases for polynomial (K,N,B)
        '''
        pass
    
    def get_coeffs(self):
        '''return coefficients for the polynomial - reparameterized if needed
        Return:
            coeffs (torch.dense): coefficients for the polynomial (K,)
        '''
        pass

    def _get_signal(self, pck_users, poss=None):
        if poss is None:
            inter = self.data_handler.inter
        else:
            inter = self.data_handler.pf_get_masked_inter(pck_users, poss)
        pck_users = pck_users.long().cpu().numpy()
        signal = torch.tensor(inter[pck_users].todense()).to(args.device).T 
        return signal # (N,B)
    
    def _get_norm_inter(self, pck_users, poss=None):
        if poss is None:
            norm_inter = self.data_handler.norm_inter
        else:
            norm_inter = self.data_handler.pf_get_masked_norm_inter(pck_users, poss)
        return norm_inter # (N,N)

    def _get_ideal(self): 
        return self.data_handler.ideal # (N,ideal_num)

    def forward(self, pck_users, poss=None):        
        signal = self._get_signal(pck_users, poss) # (N,B)
        norm_inter = self._get_norm_inter(pck_users, poss) # (N,N)
        bases = self.get_bases(signal, norm_inter) # (K,N,B)
        coeffs = self.get_coeffs() # (K,)
        full_preds = torch.einsum('K,KNB->BN', coeffs, bases) # (B,N)
        if args.ideal_num:
            ideal = self._get_ideal() # (N,ideal_num)
            full_preds += args.ideal_weight * (signal.T @ ideal @ ideal.T)
        return full_preds # (B,N)

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        full_preds = self.forward(pck_users)
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds