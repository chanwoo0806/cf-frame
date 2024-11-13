import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from cf_frame.model import BaseModel
from cf_frame.configurator import args

class GSP_Norm(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.inter = data_handler.trn_mat  # R (scipy-coo)
        self.a = args.a
        self.b = args.b
        self.c = args.c
        self.set_filter()
        
    def set_filter(self):
        user_degree = np.array(self.inter.sum(axis=1)).flatten()  # Du
        item_degree = np.array(self.inter.sum(axis=0)).flatten()  # Di
        
        self.user_deg_a = sp.diags(np.power(user_degree + 1e-10, -self.a))  # Du^(-a)
        self.item_deg_a = sp.diags(np.power(item_degree + 1e-10, -self.a))  # Di^(-a)

        self.user_deg_b = sp.diags(np.power(user_degree + 1e-10, -self.b))  # Du^(-b)
        self.item_deg_b = sp.diags(np.power(item_degree + 1e-10, -self.b))  # Di^(-b)

        self.user_deg_c = sp.diags(np.power(user_degree + 1e-10, -self.c))  # Du^(-c)
        self.item_deg_c = sp.diags(np.power(item_degree + 1e-10, -self.c))  # Di^(-c)

        self.inter = self.inter.tocsr() # R (scipy-csr)
        self.inter_t = self.inter.transpose() # R^T (scipy-csc)
        
        gram_item = self.inter_t @ self.inter
        gram_item_degree = np.array(gram_item.sum(axis=1)).flatten()
        self.gram_item_d_inv = sp.diags(np.power(gram_item_degree + 1e-10, -1))
        self.gram_item_d_inv_sqrt = sp.diags(np.power(gram_item_degree + 1e-10, -0.5))
        
    
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long().cpu().numpy()
        
        signal = self.inter[pck_users].todense() # rows of R (numpy)

        '''
            (1) D_I^{-a} R^t D_U^{-b} R D_I^{-c}
            (2) D_U^{-a} R D_I^{-b} R^t D_U^{-c}
        '''
        # (1)
        full_preds = signal @ self.item_deg_a @ self.inter_t @ self.user_deg_b @ self.inter.tocsc() @ self.item_deg_c

        # (2)
        # full_preds = signal @ self.user_deg_a @ self.inter.tocsc() @ self.item_deg_b @ self.inter_t @ self.item_deg_c

        full_preds = torch.tensor(full_preds).to(args.device)    
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
        