import torch
import numpy as np
from cf_frame.model import BaseModel
from cf_frame.configurator import args
import gc
from tqdm import tqdm 


class SGFCF(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.freq_matrix = self._convert_sp_mat_to_sp_tensor(data_handler.get_inter()).to_dense().to(args.device)
        print(type(self.freq_matrix))

        self.get_homo_ratio()
        self.set_filter()

    def get_homo_ratio(self) -> None:
        freq_matrix = self.freq_matrix
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

        # No return values
        self.homo_ratio_user = torch.Tensor(homo_ratio_user).to(args.device)
        self.homo_ratio_item = torch.Tensor(homo_ratio_item).to(args.device)
        del homo_ratio_user, homo_ratio_item

    def set_filter(self) -> None:
        freq_matrix = self.freq_matrix
        k = args.k
        eps = args.eps
        alpha = args.alpha

        D_u = 1 / (freq_matrix.sum(1) + alpha).pow(eps)
        D_i = 1 / (freq_matrix.sum(0) + alpha).pow(eps)

        D_u[D_u == float('inf')] = 0
        D_i[D_i == float('inf')] = 0

        norm_freq_matrix = D_u.unsqueeze(1) * freq_matrix * D_i
        del freq_matrix

        U, value, V = torch.svd_lowrank(norm_freq_matrix, q=k+200, niter=30)
        value = value / value.max()

        # No return values
        self.U = U; del U
        self.V = V; del V
        self.value = value; del value
        self.norm_freq_matrix = norm_freq_matrix; del norm_freq_matrix

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data

        gamma = args.gamma
        k = args.k

        U = self.U[pck_users, :]
        homo_ratio_user = self.homo_ratio_user[pck_users]
        freq_matrix = self.freq_matrix[pck_users, :]
        norm_freq_matrix = self.norm_freq_matrix[pck_users, :]
        
        V = self.V
        homo_ratio_item = self.homo_ratio_item
        value = self.value

        user_weights = self._individual_weight(value[:k], homo_ratio_user)
        item_weights = self._individual_weight(value[:k], homo_ratio_item)

        rate_matrix = (U[:, :k] * user_weights).mm((V[:, :k] * item_weights).t())
        del homo_ratio_user, homo_ratio_item
        gc.collect()
        torch.cuda.empty_cache()

        rate_matrix = rate_matrix / (rate_matrix.sum(1, keepdim=True) + 1e-8)  # Avoid division by zero
        norm_freq_matrix = norm_freq_matrix.mm(norm_freq_matrix.t()).mm(norm_freq_matrix)
        norm_freq_matrix = norm_freq_matrix / (norm_freq_matrix.sum(1, keepdim=True) + 1e-8)
        
        rate_matrix = (rate_matrix + gamma * norm_freq_matrix).sigmoid()

        rate_matrix = rate_matrix - freq_matrix * 1000
        del freq_matrix, norm_freq_matrix, U, V, value
        gc.collect()
        torch.cuda.empty_cache()

        full_preds = self._mask_predict(rate_matrix, train_mask)
        return full_preds

    def _individual_weight(self, value, homo_ratio):
        y_min = args.beta_1
        y_max = args.beta_2

        x_min = homo_ratio.min()
        x_max = homo_ratio.max()

        homo_weight = (y_max - y_min) / (x_max - x_min) * homo_ratio + (x_max * y_min - y_max * x_min) / (x_max - x_min)
        return value.pow(homo_weight.unsqueeze(1))
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))