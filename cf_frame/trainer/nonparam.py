from cf_frame.configurator import args
from cf_frame.util import log_exceptions
from cf_frame.trainer import BaseTrainer
import time

class NonParam(BaseTrainer):
    def __init__(self, data_handler, logger, loss):
        super().__init__(data_handler, logger, loss)
        
    @log_exceptions
    def train(self, model):
        start_time = time.time()
        model.set_filter()
        self.logger.log(f">>> Filter setting time: {(time.time() - start_time)/60:.1f} mins")
        self.evaluate(model)
        return model
    
    # if test needed, delete below two methods
    @log_exceptions
    def evaluate(self, model, epoch_idx=None):
        eval_result = super().evaluate(model, epoch_idx)
        if args.summary is not None:
            self.logger.log_summary(eval_result, args.ks)
        return eval_result
    
    @log_exceptions
    def test(self, model):
        pass