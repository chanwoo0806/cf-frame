import torch
import numpy as np
import scipy.sparse as sp
from cf_frame.model import BaseModel
from cf_frame.configurator import args

class GSP_Poly(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.inter = data_handler.trn_mat # R (scipy.sparse.coo)
        self.normalize = args.normalize
        self.coeffs = args.coeffs
        self.layer_num = len(args.coeffs)
        
    def set_filter(self):
        user_degree = np.array(self.inter.sum(axis=1)).flatten() # Du
        item_degree = np.array(self.inter.sum(axis=0)).flatten() # Di
        user_d_inv_sqrt = sp.diags(np.power(user_degree + 1e-10, -0.5)) # Du^(-0.5)
        item_d_inv_sqrt = sp.diags(np.power(item_degree + 1e-10, -0.5)) # Di^(-0.5)
        
        if self.normalize:
            self.inter_ = (user_d_inv_sqrt @ self.inter @ item_d_inv_sqrt).tocsc() # R_tilde (scipy.sparse.csc)
            self.inter_t_ = self.inter_.transpose().tocsc() # R_tilde^T (scipy.sparse.csc)
        else:
            self.inter_ = self.inter.tocsc() # R
            self.inter_t_ = self.inter.transpose().tocsc() # R^T
        
        self.inter = self.inter.tocsr() # R (scipy.sparse.csr)
    
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long().cpu().numpy()
        
        signal = self.inter[pck_users].todense() # rows of R (numpy.ndarray)
        
        full_preds = signal @ self.inter_t_ @ self.inter_ * self.coeffs[-1]
        for l in range(1, self.layer_num):
            full_preds = full_preds @ self.inter_t_ @ self.inter_
            full_preds += signal @ self.inter_t_ @ self.inter_ * self.coeffs[-1-l]
                
        full_preds = torch.tensor(full_preds).to(args.device)    
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds