from cf_frame.configurator import args
from cf_frame.util import log_exceptions
from cf_frame.trainer import BaseTrainer
import time

class NonParam(BaseTrainer):
    def __init__(self, data_handler, logger, loss):
        super().__init__(data_handler, logger, loss)
        
    @log_exceptions
    def train(self, model):
        return model