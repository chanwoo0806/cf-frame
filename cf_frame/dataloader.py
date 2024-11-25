import os
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, coo_matrix
import torch
import torch.utils.data as data
from cf_frame.configurator import args
try:
    import cf_frame.sampling as sampling
except:
    print("No CMAKE - We can't use cpp sampling.")
from cf_frame.util import scipy_coo_to_torch_sparse


class PairwiseTrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        self.item_num = coomat.shape[1]
    
    def sample_negs(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(self.item_num)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg
    
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class PointwiseTrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        
    def sample_negs(self):
        pass
    
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx]


class AllRankTstData(data.Dataset):
    def __init__(self, coomat, trn_mat):
        self.csrmat = (trn_mat.tocsr() != 0) * 1.0
        user_pos_lists = [list() for i in range(coomat.shape[0])]
        test_users = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            user_pos_lists[row].append(col)
            test_users.add(row)
        self.test_users = np.array(list(test_users))
        self.user_pos_lists = user_pos_lists
    
    def __len__(self):
        return len(self.test_users)
    
    def __getitem__(self, idx):
        pck_user = self.test_users[idx]
        pck_mask = self.csrmat[pck_user].toarray()
        pck_mask = np.reshape(pck_mask, [-1])
        return pck_user, pck_mask


class MultiNegTrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = None
        self.item_num = coomat.shape[1]

        interacted_items = [list() for i in range(coomat.shape[0])]
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            interacted_items[row].append(col)
        self.interacted_items = interacted_items

    def sample_negs(self):
        item_num = self.item_num  # Dataset에 의해서 알아서 결정됨
        neg_ratio = args.negative_num  # Number of negative samples for each user, pos pairs.
        sampling_sift_pos = args.sampling_sift_pos  # True -> 이미 상호작용한 아이템을 Negative samping에서 제외
        interacted_items = self.interacted_items
        neg_candidates = np.arange(item_num)
        
        if sampling_sift_pos:
            neg_items = []
            for u in self.rows:
                probs = np.ones(item_num)
                probs[interacted_items[u]] = 0
                probs /= np.sum(probs)
                u_neg_items = np.random.choice(neg_candidates, size = neg_ratio, p = probs, replace = True).reshape(1, -1)
                neg_items.append(u_neg_items)
            neg_items = np.concatenate(neg_items, axis = 0) 
        else:
            neg_items = np.random.choice(neg_candidates, (len(self.rows), neg_ratio), replace = True)
        self.negs = torch.from_numpy(neg_items)
        print(self.negs.shape)
        assert self.negs.shape[0] == len(self.rows)
        assert self.negs.shape[1] == neg_ratio

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

      
class MultiNegTrnData_CPP(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.neg_num = args.negative_num
        self.user_num, self.item_num = coomat.shape
        interacted_items = [list() for _ in range(coomat.shape[0])]
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            interacted_items[row].append(col)
        self.interacted_items = interacted_items

    def sample_negs(self):
        self.result = sampling.sample_negative_ByUser(
            self.rows,
            self.item_num,
            self.interacted_items,
            self.neg_num
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.result[idx, 0], self.result[idx, 1], self.result[idx, 2:]


class DataHandler:
    def __init__(self, loss_type):
        predir = f'./dataset/{args.dataset}/'
        self.trn_file = predir + 'train_mat.pkl'
        self.val_file = predir + 'valid_mat.pkl'
        self.tst_file = predir + 'test_mat.pkl'
        self.loss_type = loss_type

    def _load_one_mat(self, file):
        """Load one single adjacent matrix from file
        Args:
            file (string): path of the file to load
        Returns:
            scipy.sparse.coo_matrix: the loaded adjacent matrix
        """
        with open(file, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)
        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)
        return mat
    
    ########################################################
    ### DataLoader for Training, Validating, and Testing ###
    ########################################################
    
    def load_data(self):
        trn_mat = self._load_one_mat(self.trn_file)
        tst_mat = self._load_one_mat(self.tst_file)
        val_mat = self._load_one_mat(self.val_file) if os.path.exists(self.val_file) else tst_mat
        self.user_num, self.item_num = trn_mat.shape
        
        # Load dataset
        if self.loss_type == 'pairwise':
            trn_data = PairwiseTrnData(trn_mat)
        elif self.loss_type == 'pointwise':
            trn_data = PointwiseTrnData(trn_mat)
        elif self.loss_type == 'multineg':
            trn_data = MultiNegTrnData(trn_mat)
        elif self.loss_type == 'multineg_cpp':
            trn_data = MultiNegTrnData_CPP(trn_mat)
        elif self.loss_type == 'nonparam':
            pass
        else:
            raise NotImplementedError
        val_data = AllRankTstData(val_mat, trn_mat)
        tst_data = AllRankTstData(tst_mat, trn_mat)
        
        # Set dataloader
        self.train_dataloader = data.DataLoader(trn_data, batch_size=args.trn_batch, shuffle=True, num_workers=0) if self.loss_type != 'nonparam' else None
        self.valid_dataloader = data.DataLoader(val_data, batch_size=args.tst_batch, shuffle=False, num_workers=0)
        self.test_dataloader  = data.DataLoader(tst_data, batch_size=args.tst_batch, shuffle=False, num_workers=0)

    ############################################################
    ### Interaction, Adjacency Matrix for Graph-based Models ###
    ############################################################
    
    def get_inter(self):
        return self._load_one_mat(self.trn_file)

    def get_normalized_inter(self, inter=None):
        # Get interaction matrix
        inter = self._load_one_mat(self.trn_file) if inter is None else inter # R
        # Normalize interaction matrix
        user_degree = np.array(inter.sum(axis=1)).flatten() # Du
        item_degree = np.array(inter.sum(axis=0)).flatten() # Di
        user_d_inv_sqrt = sp.diags(np.power(user_degree + 1e-10, -0.5)) # Du^(-0.5)
        item_d_inv_sqrt = sp.diags(np.power(item_degree + 1e-10, -0.5)) # Di^(-0.5)
        norm_inter = (user_d_inv_sqrt @ inter @ item_d_inv_sqrt).tocoo() # Du^(-0.5) * R * Di^(-0.5)
        return norm_inter
    
    def get_normalized_adj(self, inter=None):
        # Get adjacency matrix
        inter = self._load_one_mat(self.trn_file) if inter is None else inter # R
        zero_u = csr_matrix((self.user_num, self.user_num))
        zero_i = csr_matrix((self.item_num, self.item_num))
        adj = sp.vstack([sp.hstack([zero_u, inter]), sp.hstack([inter.transpose(), zero_i])])
        adj = (adj != 0) * 1.0 # A
        # adj = (adj + sp.eye(adj.shape[0])) * 1.0 # self-connection
        # Normalize adjacency matrix
        degree = np.array(adj.sum(axis=-1)) + 1e-10 # D
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt) # D^(-0.5)
        norm_adj = adj.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo() # D^(-0.5) * A * D^(-0.5)
        return norm_adj

    ##################################################
    ### Methods for Graph Polynomial Filter Models ###
    ##################################################

    def pf_set_inter(self):
        self.inter = self.get_inter().tocsr() # scipy.sparse.csr
        norm_inter = self.get_normalized_inter()
        self.norm_inter = scipy_coo_to_torch_sparse(norm_inter) # torch.sparse.FloatTensor
        if 'mask' in args.loss.lower():
            self.inter_dok = self.get_inter().todok() # scipy.sparse.dok
            self.norm_inter_dok = self.get_normalized_inter().todok() # scipy.sparse.dok
    
    def pf_set_ideal(self):
        path = f'./dataset/{args.dataset}/svd/v_full.npy'
        if os.path.exists(path):
            v = np.load(path)[:, :args.ideal_num] # caveat: singular vectors need to be sorted in ascending order
            self.ideal = torch.tensor(v).to(args.device)
        else:
            norm_inter = self.get_normalized_norm_inter()
            u, s, v = svds(norm_inter, which='LM', k=args.ideal_num, random_state=args.rand_seed) # SVD for k largest singular vals
            self.ideal = torch.tensor(v.T).to(args.device)
            
    def pf_get_masked_inter(self, row_indices, col_indices):
        return self._mask_mat(self.inter_dok, row_indices, col_indices, rescale=False).tocsr() # scipy.sparse.csr
    
    def pf_get_masked_norm_inter(self, row_indices, col_indices, renormalize=True):
        if renormalize:
            norm_inter = self._mask_mat(self.inter_dok, row_indices, col_indices, rescale=False)
            norm_inter = self.get_normalized_inter(norm_inter)
        else:
            norm_inter = self._mask_mat(self.norm_inter_dok, row_indices, col_indices, rescale=True)
        return scipy_coo_to_torch_sparse(norm_inter) # torch.sparse.FloatTensor      

    def _mask_mat(self, mat, row_indices, col_indices, rescale):
        '''
        Args:
            mat: scipy.sparse.dok_matrix
            row_indices: list of row indices to mask (torch.IntTensor)
            col_indices: list of column indices to mask (torch.IntTensor)
        Returns:
            scipy.sparse.coo_matrix
        '''
        before = mat.nnz  
        for row_idx, col_idx in zip(row_indices.tolist(), col_indices.tolist()):
            mat[(row_idx, col_idx)] = 0.0
        after = mat.nnz
        mat = mat.tocoo()
        if rescale:
            keep_ratio = after / before
            mat /= keep_ratio
        return mat