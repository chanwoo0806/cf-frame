import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from cf_frame.configurator import args
from cf_frame.trainer import BaseTrainer
from cf_frame.util import log_exceptions, cheby, bern

class PolyFilter(BaseTrainer):
    def __init__(self, data_handler, logger, loss):
        super().__init__(data_handler, logger, loss)

    @log_exceptions
    def train(self, model):
        self._analyze(model, 'initial')
        if args.nonparam:
            self.evaluate(model)
            best_model = model
        else:
            best_model = super().train(model)
        return best_model

    @log_exceptions
    def evaluate(self, model, epoch_idx=None):
        eval_result = super().evaluate(model, epoch_idx)
        epoch_idx = epoch_idx if epoch_idx is not None else 'best'
        self._analyze(model, f'epoch_{epoch_idx}')
        return eval_result

    @log_exceptions
    def _analyze(self, model, name):
        with torch.no_grad():
            # Log filter params    
            self.logger.log(f'[Params] {str_params(model.params)}')
            # Draw filter figs
            if not os.path.exists(f'{args.path}/figs'):
                os.makedirs(f'{args.path}/figs')
            x, bases = get_bases(model.type, args.order)
            save_filter_fig(name, x, bases, model.get_coeffs())


def str_params(params):
    params = params.tolist()
    params = [f'({i}) {w:.3f}' for i, w in enumerate(params)]
    params = ' '.join(params)
    return params

def get_bases(poly, order):
    if poly == 'mono':
        f = lambda k, x: x**k
        interval = (0,1)
    elif poly == 'cheb':
        f = lambda k, x: cheby(k, x)
        interval = (-1,1)
    elif poly == 'bern':
        f = lambda k, x: bern(order, k, x)
        interval = (0,1)
    bases = []
    x = np.linspace(*interval, 100)
    for k in range(order+1):
        bases.append(f(k,x))
    y = np.stack(bases, axis=1)
    return x, y

def save_filter_fig(name, x, bases, weights):
    weights = weights.cpu().numpy()
    f = np.dot(bases, weights)
    b = bases * weights
    weights = [f'[{i}] {w:.2f}' for i, w in enumerate(weights)]
    # Full Function
    plt.plot(x, f)
    plt.axhline(0, color='black', lw=1.5)
    plt.grid(True)
    plt.xlabel('lambda')
    plt.title(' '.join(weights))
    plt.savefig(f'{args.path}/figs/{name}_full.png')
    plt.clf()
    # Each Basis
    plt.plot(x, b)
    plt.axhline(0, color='black', lw=1.5)
    plt.grid(True)
    plt.xlabel('lambda')
    plt.legend(weights)
    plt.savefig(f'{args.path}/figs/{name}_each.png')
    plt.clf()