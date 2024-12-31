import torch
from cf_frame.model import BaseModel
from cf_frame.configurator import args


class TurboCF(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.R_tr = self._convert_sp_mat_to_sp_tensor(data_handler.get_inter()).coalesce().to(args.device)
        self.set_filter()

    def set_filter(self):
        R_norm = self._normalize_sparse_adjacency_matrix(self.R_tr, args.alpha).to(args.device)
        if args.dense:
            R_norm = R_norm.to_dense()
            self.P = R_norm.T @ R_norm
            self.P.data **= args.power            
        else:
            self.P = torch.sparse.mm(R_norm.T, R_norm).coalesce()
            self.P._values().pow_(args.power)
        del R_norm

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        R = self.R_tr.index_select(0, pck_users)
        
        if args.dense:
            R = R.to_dense()
            if args.filter == 1:
                results = R @ (self.P)  
            elif args.filter == 2:
                results = R @ (2 * self.P - self.P @ self.P)
            elif args.filter == 3:
                results = R @ (self.P + 0.01 * (-self.P @ self.P @ self.P + 10 * self.P @ self.P - 29 * self.P))

            rate_matrix = results + (-99999) * R
            full_preds = self._mask_predict(rate_matrix, train_mask)
            
        else:
            if args.filter == 1:
                results = torch.sparse.mm(R, self.P)
            elif args.filter == 2:
                PP = torch.sparse.mm(self.P, self.P)
                results = torch.sparse.mm(R, 2 * self.P - PP)
            elif args.filter == 3:
                PP = torch.sparse.mm(self.P, self.P)
                PPP = torch.sparse.mm(PP, self.P)
                results = torch.sparse.mm(R, self.P + 0.01 * (-PPP + 10 * PP - 29 * self.P))

            rate_matrix = results + (-99999) * R
            del results, R
            full_preds = self._mask_predict(rate_matrix.to_dense(), train_mask)

        return full_preds

    def _normalize_sparse_adjacency_matrix(self, adj_matrix, alpha):
        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        col_sum = torch.sparse.sum(adj_matrix, dim=0).to_dense()

        row_inv = torch.pow(row_sum, -alpha).to(adj_matrix.device)
        col_inv = torch.pow(col_sum, alpha - 1).to(adj_matrix.device)

        row_diag = torch.sparse_coo_tensor(
            torch.arange(row_inv.size(0)).repeat(2, 1).to(adj_matrix.device),
            row_inv,
            (row_inv.size(0), row_inv.size(0)),
        ).coalesce()

        col_diag = torch.sparse_coo_tensor(
            torch.arange(col_inv.size(0)).repeat(2, 1).to(adj_matrix.device),
            col_inv,
            (col_inv.size(0), col_inv.size(0)),
        ).coalesce()

        norm_adj = torch.sparse.mm(row_diag, torch.sparse.mm(adj_matrix, col_diag))
        return norm_adj.coalesce()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        index = torch.stack([row, col])
        data = torch.tensor(coo.data, dtype=torch.float)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape)).coalesce()
