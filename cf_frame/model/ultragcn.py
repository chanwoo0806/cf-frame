import numpy as np
import torch
from torch import nn
from cf_frame.model import BaseModel
from cf_frame.configurator import args

init = nn.init.xavier_uniform_


class UltraGCN(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.embed_dim = args.embed_dim
        self.user_embeds = nn.Parameter(init(torch.empty(self.user_num, self.embed_dim)))
        self.item_embeds = nn.Parameter(init(torch.empty(self.item_num, self.embed_dim)))

        self.train_mat = data_handler.get_inter().todok()  # scipy.sparse.dok_matrix
        self.constraint_mat: dict = self.get_constraint_matrix()
        self.ii_neighbor_mat, self.ii_constraint_mat = self.get_ii_constraint_mat()

    def get_constraint_matrix(self) -> dict:
        items_D = np.sum(self.train_mat, axis = 0).reshape(-1)
        users_D = np.sum(self.train_mat, axis = 1).reshape(-1)
        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
        constraint_mat = {
            "beta_uD": torch.from_numpy(beta_uD).reshape(-1).to(args.device),
            "beta_iD": torch.from_numpy(beta_iD).reshape(-1).to(args.device)
        }
        return constraint_mat

    def get_ii_constraint_mat(self):
        print('Computing \\Omega for the item-item graph... ')

        num_neighbors = args.num_neighbors
        ii_diagonal_zero = args.ii_diagonal_zero

        A = self.train_mat.T.dot(self.train_mat)
        n_items = A.shape[0]
        res_mat = torch.zeros((n_items, num_neighbors))
        res_sim_mat = torch.zeros((n_items, num_neighbors))
        if ii_diagonal_zero:
            A[range(n_items), range(n_items)] = 0
        items_D = np.sum(A, axis = 0).reshape(-1)
        users_D = np.sum(A, axis = 1).reshape(-1)

        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
        all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
        for i in range(n_items):
            row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
            row_sims, row_idxs = torch.topk(row, num_neighbors)
            res_mat[i] = row_idxs
            res_sim_mat[i] = row_sims
            if i % 15000 == 0:
                print('i-i constraint matrix {} ok'.format(i))

        print('Computation \\Omega OK!')
        return res_mat.long().to(args.device), res_sim_mat.float().to(args.device)

    def forward(self):
        return self.user_embeds, self.item_embeds

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward()
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
    