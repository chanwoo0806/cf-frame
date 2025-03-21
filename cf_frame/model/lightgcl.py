import torch as t
import numpy as np
from torch import nn

from cf_frame.model import BaseModel
from cf_frame.configurator import args
from cf_frame.module import SvdDecomposition

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class LightGCL(BaseModel):
    def __init__(self, data_handler):
        super(LightGCL, self).__init__(data_handler)

        train_mat = data_handler._load_one_mat(data_handler.trn_file)
        rowD = np.array(train_mat.sum(1)).squeeze()
        colD = np.array(train_mat.sum(0)).squeeze()
        for i in range(len(train_mat.data)):
            train_mat.data[i] = train_mat.data[i] / pow(rowD[train_mat.row[i]] * colD[train_mat.col[i]], 0.5)
        adj_norm = self._scipy_sparse_mat_to_torch_sparse_tensor(train_mat)
        self.adj = adj_norm.coalesce().cuda()

        self.svd_decompose = SvdDecomposition(svd_q=args.svd_q)
        self.ut, self.vt, self.u_mul_s, self.v_mul_s = self.svd_decompose(self.adj)

        self.dropout = args.dropout
        self.layer_num = args.layer_num
        self.embedding_size = args.embed_dim

        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.E_u_list = [None] * (self.layer_num+1)
        self.E_i_list = [None] * (self.layer_num+1)
        self.E_u_list[0] = self.user_embeds
        self.E_i_list[0] = self.item_embeds
        self.Z_u_list = [None] * (self.layer_num+1)
        self.Z_i_list = [None] * (self.layer_num+1)
        self.G_u_list = [None] * (self.layer_num+1)
        self.G_i_list = [None] * (self.layer_num+1)
        self.G_u_list[0] = self.user_embeds
        self.G_i_list[0] = self.item_embeds
        self.E_u = None
        self.E_i = None
        self.act = nn.LeakyReLU(0.5)
        self.Ws = nn.ModuleList([W_contrastive(self.embedding_size) for i in range(self.layer_num)])
        self.is_training = True

    def _scipy_sparse_mat_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = t.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = t.from_numpy(sparse_mx.data)
        shape = t.Size(sparse_mx.shape)
        return t.sparse.FloatTensor(indices, values, shape)

    def _spmm(self,sp, emb):
        sp = sp.coalesce()
        cols = sp.indices()[1]
        rows = sp.indices()[0]
        col_segs = emb[cols] * t.unsqueeze(sp.values(),dim=1)
        result = t.zeros((sp.shape[0],emb.shape[1])).cuda()
        result.index_add_(0, rows, col_segs)
        return result

    def _sparse_dropout(self, mat, dropout):
        indices = mat.indices()
        values = nn.functional.dropout(mat.values(), p=dropout)
        size = mat.size()
        return t.sparse.FloatTensor(indices, values, size)

    def forward(self, test=False):
        if test and self.E_u is not None:
            return self.E_u, self.E_i
        for layer in range(1, self.layer_num+1):
            # GNN propagation
            self.Z_u_list[layer] = self._spmm(self._sparse_dropout(self.adj, self.dropout), self.E_i_list[layer-1])
            self.Z_i_list[layer] = self._spmm(self._sparse_dropout(self.adj, self.dropout).transpose(0,1), self.E_u_list[layer-1])

            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer-1]
            self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
            ut_eu = self.ut @ self.E_u_list[layer-1]
            self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]  # + self.E_u_list[layer-1]
            self.E_i_list[layer] = self.Z_i_list[layer]  # + self.E_i_list[layer-1]

        # aggregate across layers
        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)

        return self.E_u, self.E_i

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(test=True)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds


class W_contrastive(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(t.empty(d,d)))

    def forward(self,x):
        return x @ self.W