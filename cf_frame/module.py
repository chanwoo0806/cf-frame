import torch
from torch import nn

class EdgeDrop(nn.Module):
    """ Drop edges in a graph.
    """
    def __init__(self, resize_val=False):
        super().__init__()
        self.resize_val = resize_val

    def forward(self, adj, keep_rate):
        """
        :param adj: torch_adj in data_handler
        :param keep_rate: ratio of preserved edges
        :return: adjacency matrix after dropping edges
        """
        if keep_rate == 1.0: return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = (torch.rand(edgeNum) + keep_rate).floor().type(torch.bool)
        newVals = vals[mask] / (keep_rate if self.resize_val else 1.0)
        newIdxs = idxs[:, mask]
        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
    

class SvdDecomposition(nn.Module):
    """ Utilize SVD to decompose matrix (used in LightGCL)
    """
    def __init__(self, svd_q):
        super(SvdDecomposition, self).__init__()
        self.svd_q = svd_q

    def forward(self, adj):
        """
        :param adj: torch sparse matrix
        :return: matrices obtained by SVD decomposition
        """
        svd_u, s, svd_v = torch.svd_lowrank(adj, q=self.svd_q)
        u_mul_s = svd_u @ torch.diag(s)
        v_mul_s = svd_v @ torch.diag(s)
        del s
        return svd_u.T, svd_v.T, u_mul_s, v_mul_s
        