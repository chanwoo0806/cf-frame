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