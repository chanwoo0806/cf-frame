import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from cf_frame.model import BaseModel
from cf_frame.configurator import args

class GFCF(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.inter = data_handler.trn_mat # R (scipy-coo)
        self.cutoff = 256
        
    def set_filter(self):
        user_degree = np.array(self.inter.sum(axis=1)).flatten() # Du
        item_degree = np.array(self.inter.sum(axis=0)).flatten() # Di
        user_d_inv_sqrt = sp.diags(np.power(user_degree + 1e-10, -0.5)) # Du^(-0.5)
        item_d_inv_sqrt = sp.diags(np.power(item_degree + 1e-10, -0.5)) # Di^(-0.5)
        item_d_sqrt = sp.diags(np.power(item_degree, 0.5))  # Di^(0.5)
        self.item_d_sqrt = item_d_sqrt
        self.item_d_inv_sqrt = item_d_inv_sqrt
        
        self.norm_inter = (user_d_inv_sqrt @ self.inter @ item_d_inv_sqrt).tocsc() # R_tilde (scipy-csc)
        self.norm_inter_t = self.norm_inter.transpose().tocsc() # R_tilde^T (scipy-csc)

        u, s, v = svds(self.norm_inter, which='LM', k=self.cutoff, random_state=args.rand_seed) # SVD for k largest singular values
        self.v = v.T # right singular vecs (numpy)
        
        self.inter = self.inter.tocsr() # R (scipy-csr)
    
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long().cpu().numpy()
        
        signal = self.inter[pck_users].todense() # rows of R (numpy)
        pred_linear = signal @ self.norm_inter_t @ self.norm_inter
        pred_ideal  = signal @ self.item_d_inv_sqrt @ self.v @ self.v.T @ self.item_d_sqrt
        
        full_preds = pred_linear + args.ideal * pred_ideal
        full_preds = torch.tensor(full_preds).to(args.device)    
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
        