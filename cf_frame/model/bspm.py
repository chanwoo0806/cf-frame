import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from cf_frame.model import BaseModel
from cf_frame.configurator import args
from torchdiffeq import odeint


class BSPM(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.adj_mat = data_handler.trn_mat.tolil()  

        self.device = args.device

        self.idl_solver = args.solver_idl
        self.blur_solver = args.solver_blr
        self.sharpen_solver = args.solver_shr

        self.idl_beta = args.idl_beta
        self.factor_dim = args.factor_dim

        idl_T     = args.T_idl ; idl_K     = args.K_idl
        blur_T    = args.T_b   ; blur_K    = args.K_b
        sharpen_T = args.T_s   ; sharpen_K = args.K_s

        self.idl_times        = torch.linspace(0, idl_T, idl_K+1).float().to(self.device)
        self.blurring_times   = torch.linspace(0, blur_T, blur_K+1).float().to(self.device)
        self.sharpening_times = torch.linspace(0, sharpen_T, sharpen_K+1).float().to(self.device)

        self.final_sharpening = args.final_sharpening
        self.sharpening_off = args.sharpening_off
        self.t_point_combination = args.t_point_combination

        print("idl time: ",self.idl_times)
        print("blur time: ",self.blurring_times)
        print("sharpen time: ",self.sharpening_times)

    def set_filter(self):
        adj_mat = self.adj_mat
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        del norm_adj, d_mat
        if args.dataset != 'amazon-book':
            u, s, self.vt = svds(self.norm_adj, k=self.factor_dim)
            del u
            del s

        linear_Filter = self.norm_adj.T @ self.norm_adj
        self.linear_Filter = self.convert_sp_mat_to_sp_tensor(linear_Filter).to_dense().to(self.device)

        if args.dataset != 'amazon-book':
            left_mat = self.d_mat_i @ self.vt.T
            right_mat = self.vt @ self.d_mat_i_inv
            self.left_mat, self.right_mat = torch.FloatTensor(left_mat).to(self.device), torch.FloatTensor(right_mat).to(self.device)

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long().cpu().numpy()

        adj_mat = self.adj_mat
        if not torch.is_tensor(adj_mat):
            adj_mat = self.convert_sp_mat_to_sp_tensor(self.adj_mat).to_dense()

        ds_name = args.dataset
        batch_test = adj_mat[pck_users, :].to(self.device).to_sparse()

        with torch.no_grad():
            if(ds_name != 'amazon-book'):
                idl_out = torch.mm(batch_test, self.left_mat @ self.right_mat)

            if(ds_name != 'amazon-book'):
                blurred_out = torch.mm(batch_test, self.linear_Filter)
            else:
                blurred_out = torch.mm(batch_test.to_dense(), self.linear_Filter)

            del batch_test
            
            if self.sharpening_off == False:
                if self.final_sharpening == True:
                    if(ds_name != 'amazon-book'):
                        sharpened_out = odeint(func=self.sharpenFunction, y0=self.idl_beta*idl_out+blurred_out, t=self.sharpening_times, method=self.sharpen_solver)
                    elif(ds_name == 'amazon-book'):
                        sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out, t=self.sharpening_times, method=self.sharpen_solver)
                else:
                    sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out, t=self.sharpening_times, method=self.sharpen_solver)
        
        if self.t_point_combination == True:
            if self.sharpening_off == False:
                U_2 =  torch.mean(torch.cat([blurred_out.unsqueeze(0),sharpened_out[1:,...]],axis=0),axis=0)
            else:
                U_2 =  blurred_out
                del blurred_out
        else:
            if self.sharpening_off == False:
                U_2 = sharpened_out[-1]
                del sharpened_out
            else:
                U_2 = blurred_out
                del blurred_out
        
        if(ds_name == 'amazon-book'):
            ret = U_2
            del U_2
        else:
            if self.final_sharpening == True:
                if self.sharpening_off == False:
                    ret = U_2
                elif self.sharpening_off == True:
                    ret = self.idl_beta * idl_out + U_2
            else:
                ret = self.idl_beta * idl_out + U_2

        full_preds = self._mask_predict(ret, train_mask)
        return full_preds
    
    def sharpenFunction(self, t, r):
        out = r @ self.linear_Filter
        return -out

    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))