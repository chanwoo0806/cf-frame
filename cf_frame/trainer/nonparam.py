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
    
    # if test needed, delete this method
    @log_exceptions
    def test(self, model):
        pass