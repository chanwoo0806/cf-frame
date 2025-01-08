import os
import torch
import random
import logging
import importlib
import numpy as np
import pickle
from cf_frame.configurator import args
from scipy.special import comb

def init_seed():
    if args.rand_seed is not None:
        seed = args.rand_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
def build_model(data_handler):
    module_path = f'cf_frame.model.{args.model}'
    module = importlib.import_module(module_path)
    for attr in dir(module):
        if attr.lower() == args.model.lower():
            return getattr(module, attr)(data_handler)
    raise NotImplementedError(f'No model named {args.model} in {module_path}')

def build_loss():
    module_path = f'cf_frame.loss'
    module = importlib.import_module(module_path)
    for attr in dir(module):
        if attr.lower() == args.loss.lower():
            return getattr(module, attr)()
    raise NotImplementedError(f'No loss named {args.loss} in {module_path}')
    
def build_trainer(data_handler, logger, loss):
    if args.trainer is None:
        module_path = 'cf_frame.trainer'
        module = importlib.import_module(module_path)
        return getattr(module, 'BaseTrainer')(data_handler, logger, loss)
    else:
        module_path = f'cf_frame.trainer.{args.trainer}'
        module = importlib.import_module(module_path)
        for attr in dir(module):
            if attr.lower() == args.trainer.lower():
                return getattr(module, attr)(data_handler, logger, loss)
        raise NotImplementedError(f'No trainer named {args.trainer} in {module_path}')

def log_exceptions(func):
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('train_logger')
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e
    return wrapper

class Logger:
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO) 
        # Print log to both file and console
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        file_handler = logging.FileHandler(f'{args.path}/train.log', 'a', encoding='utf-8')
        strm_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s')
        for handler in (file_handler, strm_handler):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.log_config()

    def log(self, message):
        self.logger.info(message)
      
    def log_config(self):
        config = '\n[CONFIGS]\n'
        for arg, value in args.__dict__.items():
            if value is not None:
                config += f'>>> {arg}: {value}\n'
        self.logger.info(config)

    def log_loss(self, epoch_idx, loss_log_dict):    
        message = f'[Epoch {epoch_idx:>4d} / {args.epoch:>4d}] '
        for loss, value in loss_log_dict.items():
            message += f'{loss}: {value:.4f} '
        self.logger.info(message)

    def log_eval(self, eval_result, ks, epoch_idx=None):
        message = '\n'
        if epoch_idx is not None:
            message += f'[Epoch {epoch_idx:>4d} / {args.epoch:>4d}]\n'
        header, values = '', ''
        for metric, result in eval_result.items():
            for i, k in enumerate(ks):
                metric_name = f'{metric}@{k}'
                header += f'{metric_name:>16s}'
                values += f'{result[i]:>16.4f}'
        message += (header + '\n' + values)
        self.logger.info(message)
        
    def log_summary(self, eval_result, ks):
        with open(f'./log/{args.summary}.csv', 'a') as f:
            message = f'{args.comment},'
            for result in eval_result.values():
                for i in range(len(ks)):
                    message += f'{result[i]:.4f},'
            f.write(message[:-1] + '\n')
     
class DisabledSummaryWriter:
    def __init__(*args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, *args, **kwargs):
        return self
    

def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res


def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))


def scipy_coo_to_torch_sparse(mat):
    # scipy.sparse.coo_matrix -> torch.sparse.FloatTensor
    idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = torch.from_numpy(mat.data.astype(np.float32))
    shape = torch.Size(mat.shape)
    return torch.sparse_coo_tensor(idxs, vals, shape, dtype=torch.float32, device=args.device)


def cheby(k, x):
    if k == 0:
        return 1 if not isinstance(x, np.ndarray) else np.ones_like(x)
    elif k == 1:
        return x
    else:
        return 2 * x * cheby(k-1, x) - cheby(k-2, x)


def bern(K, k, x):
    return comb(K, k) * (x**k) * ((1-x)**(K-k))