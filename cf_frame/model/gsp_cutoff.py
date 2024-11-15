import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from cf_frame.model import BaseModel
from cf_frame.configurator import args

class GSP_Cutoff(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.inter = data_handler.trn_mat # R (scipy.sparse.coo)
        self.normalize = args.normalize
        self.freq_num = args.freq_num
        self.filter = args.filter
        self.hyp = args.hyp
        
    def set_filter(self):
        user_degree = np.array(self.inter.sum(axis=1)).flatten() # Du
        item_degree = np.array(self.inter.sum(axis=0)).flatten() # Di
        user_d_inv_sqrt = sp.diags(np.power(user_degree + 1e-10, -0.5)) # Du^(-0.5)
        item_d_inv_sqrt = sp.diags(np.power(item_degree + 1e-10, -0.5)) # Di^(-0.5)
        
        if self.normalize:
            inter_ = (user_d_inv_sqrt @ self.inter @ item_d_inv_sqrt).tocsc() # R_tilde (scipy.sparse.csc)
        else:
            inter_ = self.inter
        u, s, v = svds(inter_, which='LM', k=self.freq_num, random_state=args.rand_seed) # SVD for k largest singular values
        self.v = v.T # right singular vecs (numpy.ndarray)
        
        x = 1 - s # normalized laplacian = identity - normalized adjacency
        if self.filter == 'exponential':
            self.filter_weight = sp.diags(np.exp(self.hyp * (-x)))
        elif self.filter == 'harmonic':
            self.filter_weight = sp.diags(1/(1 + self.hyp * x))
        elif self.filter is None:
            self.filter_weight = sp.eye(self.freq_num)
        
        self.inter = self.inter.tocsr() # R (scipy.sparse.csr)
    
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long().cpu().numpy()
        
        signal = self.inter[pck_users].todense() # rows of R (numpy.ndarray)
        full_preds  = signal @ self.v @ self.filter_weight @ self.v.T

        full_preds = torch.tensor(full_preds).to(args.device)    
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds