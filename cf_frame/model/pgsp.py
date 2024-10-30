import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from cf_frame.model import BaseModel
from cf_frame.configurator import args

class PGSP(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.inter = data_handler.trn_mat # R (scipy-coo)
        self.beta = 0.5
        self.cutoff = 512
        
    def set_filter(self):
        user_degree = np.array(self.inter.sum(axis=1)).flatten() # Du
        item_degree = np.array(self.inter.sum(axis=0)).flatten() # Di
        user_d_inv_sqrt = sp.diags(np.power(user_degree + 1e-10, -0.5)) # Du^(-0.5)
        item_d_inv_sqrt = sp.diags(np.power(item_degree + 1e-10, -0.5)) # Di^(-0.5)
        
        self.norm_inter = (user_d_inv_sqrt @ self.inter @ item_d_inv_sqrt).tocsc() # R_tilde (scipy-csc)
        self.norm_inter_t = self.norm_inter.transpose().tocsc() # R_tilde^T (scipy-csc)
        
        self.norm_item_co = (self.norm_inter @ self.norm_inter_t).tocsr() # R_tilde @ R_tilde^T (scipy-csr)
        norm_item_co_degree = np.array(self.norm_item_co.sum(axis=1)).flatten()
        self.norm_item_d_inv = sp.diags(np.power(norm_item_co_degree + 1e-10, -self.beta))
        
        self.item_d     = sp.diags(np.power(item_degree,          self.beta)) # Di^(beta)
        self.item_d_inv = sp.diags(np.power(item_degree + 1e-10, -self.beta)) # Di^(-beta)
        
        u, s, v = svds(self.norm_inter, which='LM', k=self.cutoff, random_state=args.rand_seed)
        self.u, self.v = u, v.T
        self.inter = self.inter.tocsr() # R (scipy-csr)
    
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long().cpu().numpy()
        
        signal_u = self.norm_item_co[pck_users].todense() # rows of R_tilde @ R_tilde^T (numpy) 
        pred_u_linear = signal_u @ self.norm_item_d_inv @ self.norm_inter   @ self.item_d
        pred_u_ideal  = signal_u @ self.norm_item_d_inv @ self.u @ self.v.T @ self.item_d
        
        signal_i = self.inter[pck_users].todense() # rows of R (numpy)
        pred_i_linear = signal_i @ self.item_d_inv @ self.norm_inter_t @ self.norm_inter @ self.item_d
        pred_i_ideal  = signal_i @ self.item_d_inv @ self.v @ self.v.T                   @ self.item_d
        
        full_preds =  (1 - args.ideal) * (pred_u_linear + pred_i_linear)
        full_preds += (    args.ideal) * (pred_u_ideal  + pred_i_ideal )

        full_preds = torch.tensor(full_preds).to(args.device)    
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
        