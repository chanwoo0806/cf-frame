import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from cf_frame.model import BaseModel
from cf_frame.configurator import args

class GSP_Norm(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.inter = data_handler.get_inter() # R (scipy-coo)
        # self.beta = 0.5
        # self.cutoff = 512
        self.set_filter()
        
    def set_filter(self):
        user_degree = np.array(self.inter.sum(axis=1)).flatten() # Du
        item_degree = np.array(self.inter.sum(axis=0)).flatten() # Di
        self.user_d_inv = sp.diags(np.power(user_degree + 1e-10, -1)) # Du^(-1)
        self.item_d_inv = sp.diags(np.power(item_degree + 1e-10, -1)) # Di^(-1)
        self.user_d_inv_sqrt = sp.diags(np.power(user_degree + 1e-10, -0.5)) # Du^(-0.5)
        self.item_d_inv_sqrt = sp.diags(np.power(item_degree + 1e-10, -0.5)) # Di^(-0.5)
        
        self.inter = self.inter.tocsr() # R (scipy-csr)
        self.inter_t = self.inter.transpose() # R^T (scipy-csc)
        
        # self.norm_inter = (user_d_inv_sqrt @ self.inter @ item_d_inv_sqrt).tocsc() # R_tilde (scipy-csc)
        # self.norm_inter_t = self.norm_inter.transpose().tocsc() # R_tilde^T (scipy-csc)
        
        # self.norm_item_co = (self.norm_inter @ self.norm_inter_t).tocsr() # R_tilde @ R_tilde^T (scipy-csr)
        # norm_item_co_degree = np.array(self.norm_item_co.sum(axis=1)).flatten()
        # self.norm_item_d_inv = sp.diags(np.power(norm_item_co_degree + 1e-10, -self.beta))
        
        # self.item_d     = sp.diags(np.power(item_degree,          self.beta)) # Di^(beta)
        # self.item_d_inv = sp.diags(np.power(item_degree + 1e-10, -self.beta)) # Di^(-beta)
        
        # u, s, v = svds(self.norm_inter, which='LM', k=self.cutoff, random_state=args.rand_seed)
        # self.u, self.v = u, v.T
        gram_item = self.inter_t @ self.inter
        gram_item_degree = np.array(gram_item.sum(axis=1)).flatten()
        self.gram_item_d_inv = sp.diags(np.power(gram_item_degree + 1e-10, -1))
        self.gram_item_d_inv_sqrt = sp.diags(np.power(gram_item_degree + 1e-10, -0.5))
        
    
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long().cpu().numpy()
        
        signal = self.inter[pck_users].todense() # rows of R (numpy)

        # # 111
        # full_preds = signal @ self.item_d_inv @ self.inter_t @ self.user_d_inv @ self.inter.tocsc() @ self.item_d_inv
        # # 110
        # full_preds = signal @ self.item_d_inv @ self.inter_t @ self.user_d_inv @ self.inter.tocsc()
        # # 011
        # full_preds = signal @ self.inter_t @ self.user_d_inv @ self.inter.tocsc() @ self.item_d_inv
        # # -1-
        # full_preds = signal @ self.item_d_inv_sqrt @ self.inter_t @ self.user_d_inv @ self.inter.tocsc() @ self.item_d_inv_sqrt
        # # ---
        # full_preds = signal @ self.item_d_inv_sqrt @ self.inter_t @ self.user_d_inv_sqrt @ self.inter.tocsc() @ self.item_d_inv_sqrt
        # # 000
        # full_preds = signal @ self.inter_t @ self.inter.tocsc()
        # # 001
        # full_preds = signal @ self.inter_t @ self.inter.tocsc() @ self.item_d_inv_sqrt
        # # 010
        # full_preds = signal @ self.inter_t @ self.user_d_inv @ self.inter.tocsc()
        # # 100
        # full_preds = signal @ self.item_d_inv @ self.inter_t @ self.inter.tocsc()
        # # 1--
        # full_preds = signal @ self.item_d_inv @ self.inter_t @ self.user_d_inv_sqrt @ self.inter.tocsc() @ self.item_d_inv_sqrt
        # # 11-
        # full_preds = signal @ self.item_d_inv @ self.inter_t @ self.user_d_inv @ self.inter.tocsc() @ self.item_d_inv_sqrt
        # # -10
        # full_preds = signal @ self.item_d_inv_sqrt @ self.inter_t @ self.user_d_inv @ self.inter.tocsc()
        # # -2-
        # full_preds = signal @ self.item_d_inv_sqrt @ self.inter_t @ self.user_d_inv @ self.user_d_inv @ self.inter.tocsc() @ self.item_d_inv_sqrt
        # # item_gram_degree
        # full_preds = signal @ self.gram_item_d_inv_sqrt @ self.inter_t @ self.inter.tocsc() @ self.gram_item_d_inv_sqrt
        # # item_gram_degree 10
        # full_preds = signal @ self.gram_item_d_inv @ self.inter_t @ self.inter.tocsc()
        # # item_gram_degree 01
        # full_preds = signal @ self.inter_t @ self.inter.tocsc() @ self.gram_item_d_inv
        # 101
        full_preds = signal @ self.item_d_inv @ self.inter_t @ self.inter.tocsc() @ self.item_d_inv
     
        full_preds = torch.tensor(full_preds).to(args.device)    
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
        