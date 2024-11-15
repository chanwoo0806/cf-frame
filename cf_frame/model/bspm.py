import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from cf_frame.model import BaseModel
from cf_frame.configurator import args


class BSPM(BaseModel):
    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.inter = data_handler.trn_mat  # R (scipy-coo)

    def set_filter(self):
        pass

    def full_predict(self, batch_data):
        pass