import torch
import numpy as np
from cf_frame.model import BaseModel
from cf_frame.configurator import args
import gc
from tqdm import tqdm 


class TurboCF(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.R_tr = self._convert_sp_mat_to_sp_tensor(data_handler.get_inter()).cpu().to_dense()
        self.R_norm = self._normalize_sparse_adjacency_matrix(self.R_tr, args.alpha)

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        
        R_norm = self.R_norm
        R = self.R_tr[pck_users.cpu()]

        P = R_norm.T @ R_norm
        P.data **= args.power
        
        P = P.to(device=args.device).float()
        R = R.to(device=args.device).float()

        # Our model
        if args.filter == 1:
            results = R @ (P)
        elif args.filter == 2:
            results = R @ (2 * P - P @ P)
        elif args.filter == 3:
            results = R @ (P + 0.01 * (-P @ P @ P + 10 * P @ P - 29 * P))

        # Now get the results
        rate_matrix = results + (-99999) * R
        full_preds = self._mask_predict(rate_matrix, train_mask)
        return full_preds

    def _normalize_sparse_adjacency_matrix(self, adj_matrix, alpha):
        rowsum = torch.sparse.mm(
            adj_matrix, torch.ones((adj_matrix.shape[1], 1), device=adj_matrix.device)
        ).squeeze()

        rowsum = torch.pow(rowsum, -alpha)
        
        colsum = torch.sparse.mm(
            adj_matrix.t(), torch.ones((adj_matrix.shape[0], 1), device=adj_matrix.device)
        ).squeeze()
        
        colsum = torch.pow(colsum, alpha - 1)
        
        indices = (
            torch.arange(0, rowsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
        )
        
        d_mat_rows = torch.sparse_coo_tensor(
            indices.t(), rowsum, torch.Size([rowsum.size(0), rowsum.size(0)])
        ).to(device=adj_matrix.device)
        
        indices = (
            torch.arange(0, colsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
        )
        
        d_mat_cols = torch.sparse_coo_tensor(
            indices.t(), colsum, torch.Size([colsum.size(0), colsum.size(0)])
        ).to(device=adj_matrix.device)

        norm_adj = d_mat_rows.mm(adj_matrix).mm(d_mat_cols)
        return norm_adj
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape), dtype=torch.float)