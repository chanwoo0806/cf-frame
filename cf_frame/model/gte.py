from cf_frame.model import BaseModel
import numpy as np
from cf_frame.configurator import args
import torch


class GTE(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)

        train_mat = data_handler._load_one_mat(data_handler.trn_file)

        n_u = self.user_num
        n_i = self.item_num
        k = args.layer_num

        item_rep = torch.eye(n_i).to(args.device)
        user_rep = torch.zeros(n_u, n_i).to(args.device)

        # process the adjacency matrix
        adj = self.scipy_sparse_mat_to_torch_sparse_tensor(train_mat).coalesce().to(args.device)

        # iterative representation propagation on graph
        for i in range(k):
            print("Running layer", i)
            user_rep_temp = torch.sparse.mm(adj, item_rep) + user_rep
            item_rep_temp = torch.sparse.mm(adj.transpose(0,1),user_rep) + item_rep
            user_rep = user_rep_temp
            item_rep = item_rep_temp

        # evaluation
        self.ratings = user_rep.cpu().numpy()
    
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long().cpu().numpy()
        return self._mask_predict(torch.Tensor(self.ratings[pck_users]), train_mask)

    def scipy_sparse_mat_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)