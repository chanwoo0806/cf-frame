import torch
import numpy as np
from cf_frame.model import BaseModel
from cf_frame.configurator import args
import gc
from tqdm import tqdm 
import scipy.sparse as sp
from scipy.special import expit  # for sigmoid
import scipy
from scipy.sparse.linalg import svds
from cf_frame.util import pload, pstore
from time import time


class SGFCF(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.freq_matrix = self._convert_sp_mat_to_sp_tensor(data_handler.get_inter()).to_dense().cpu()
        self.get_homo_ratio()
        self.set_filter()

    def get_homo_ratio(self):
        # If GPU is available, Use it.
        freq_matrix = self.freq_matrix.to(args.device)
        try:
            homo_ratio_user = pload(f'./dataset/{args.dataset}/homo_ratio_user.tensor')
            homo_ratio_item = pload(f'./dataset/{args.dataset}/homo_ratio_item.tensor')
        except:
            user, item = freq_matrix.shape
            homo_ratio_user, homo_ratio_item = [], []

            print('Calculate the User Homogeneity Ratio Matrix')
            for u in tqdm(range(user)):
                interactions = freq_matrix[u, :].nonzero().squeeze()
                if interactions.numel() > 1:
                    inter_items = freq_matrix[:, interactions].t()
                    inter_items[:, u] = 0
                    connect_matrix = inter_items.mm(inter_items.t())
                    size = inter_items.shape[0]
                    ratio_u=((connect_matrix!=0).sum().item()-(connect_matrix.diag()!=0).sum().item())/(size*(size-1))
                    homo_ratio_user.append(ratio_u)
                else:
                    homo_ratio_user.append(0)

            print('Calculate the Item Homogeneity Ratio Matrix')
            for i in tqdm(range(item)):
                interactions = freq_matrix[:, i].nonzero().squeeze()
                if interactions.numel() > 1:
                    inter_users = freq_matrix[interactions, :]
                    inter_users[:, i] = 0
                    connect_matrix = inter_users.mm(inter_users.t())
                    size=inter_users.shape[0]
                    ratio_i=((connect_matrix!=0).sum().item()-(connect_matrix.diag()!=0).sum().item())/(size*(size-1))
                    homo_ratio_item.append(ratio_i)
                else:
                    homo_ratio_item.append(0)

            homo_ratio_user = torch.Tensor(homo_ratio_user).to('cpu')
            homo_ratio_item = torch.Tensor(homo_ratio_item).to('cpu')

            pstore(homo_ratio_user, f'./dataset/{args.dataset}/homo_ratio_user.tensor')
            pstore(homo_ratio_item, f'./dataset/{args.dataset}/homo_ratio_item.tensor')

        self.homo_ratio_user = homo_ratio_user
        self.homo_ratio_item = homo_ratio_item
        
    def set_filter(self):
        eps = args.eps
        alpha = args.alpha

        # 1. User and Item Degree
        st = time()

        D_u = 1 / (self.freq_matrix.sum(1) + alpha).pow(eps)
        D_i = 1 / (self.freq_matrix.sum(0) + alpha).pow(eps)

        D_u[D_u == float('inf')] = 0
        D_i[D_i == float('inf')] = 0

        D_u = np.array(D_u).flatten()
        D_i = np.array(D_i).flatten()

        D_u = scipy.sparse.diags(D_u)
        D_i = scipy.sparse.diags(D_i)

        print('D_u :', D_u.shape)
        print('D_i :', D_i.shape)

        et = time()
        print(f'Degree: {et - st}sec')

        # 2. Normalized Frequency Matrix (R_tilde)
        st = time()
        freq_matrix_sp = sp.csr_matrix(self.freq_matrix.cpu().numpy())
        norm_freq_matrix = D_u @ freq_matrix_sp @ D_i
        et = time()
        print(f'Normalize: {et - st}sec')

        # 3. Singular Value Decomposition
        st = time()
        u, s, vh = svds(norm_freq_matrix, k=args.k)

        print('u:', u.shape)
        print('s:', s.shape)
        print('v:', vh.T.shape)

        self.u_k = u
        self.v_k = vh.T

        s = s / s.max()
        et = time()
        print(f'SVD: {et - st}sec')

        # 4. R_tilde @ R_tilde.T @ R_tilde
        st = time()
        norm_freq_matrix = norm_freq_matrix @ norm_freq_matrix.T @ norm_freq_matrix
        norm_freq_matrix = norm_freq_matrix / norm_freq_matrix.sum(axis=1)
        self.norm_freq_matrix = norm_freq_matrix
        et = time()
        print(f'Additional Filter Operation: {et - st}sec')

        # 5. Individual Weight
        st = time()
        self.user_weights = self._individual_weight(s, self.homo_ratio_user.numpy())
        self.item_weights = self._individual_weight(s, self.homo_ratio_item.numpy())
        et = time()
        print(f'User, Item Weights: {et - st}sec')

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long().cpu().numpy()
        
        # Perform rate matrix computation using NumPy
        rate_matrix = (self.u_k[pck_users, :] * self.user_weights[pck_users, :]) @ (self.v_k * self.item_weights).T
        
        # Normalize rows
        row_sums = rate_matrix.sum(axis=1, keepdims=True)
        rate_matrix = rate_matrix / row_sums
        
        # Add normalized frequency matrix and apply sigmoid
        rate_matrix += args.gamma * self.norm_freq_matrix[pck_users, :]
        rate_matrix = expit(rate_matrix)
        
        # Apply mask and modify based on frequency matrix
        result = torch.tensor(rate_matrix).to(args.device) - self.freq_matrix[pck_users, :].to(args.device) * 1000
        full_preds = self._mask_predict(result, train_mask)
        return full_preds

    def _individual_weight(self, value, homo_ratio):
        y_min = args.beta_1
        y_max = args.beta_2

        x_min = homo_ratio.min()
        x_max = homo_ratio.max()

        homo_weight = (y_max - y_min) / (x_max - x_min) * homo_ratio + (x_max * y_min - y_max * x_min) / (x_max - x_min)
        
        # NumPy version of `value.pow(homo_weight.unsqueeze(1))`
        homo_weight = np.expand_dims(homo_weight, axis=1)  # (N, 1) for broadcasting
        result = np.power(value, homo_weight)
        return result
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape), dtype=torch.float)
    