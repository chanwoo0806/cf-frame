import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from cf_frame.metric import Metric
from cf_frame.configurator import args
from cf_frame.util import build_model, log_exceptions, DisabledSummaryWriter

if args.tensorboard:
    writer = SummaryWriter(log_dir=args.path)
else:
    writer = DisabledSummaryWriter()

class BaseTrainer:
    def __init__(self, data_handler, logger, loss):
        self.data_handler = data_handler
        self.logger = logger
        self.loss = loss
        self.metric = Metric()

    def create_optimizer(self, model):
        self.optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs() # newly sample negs per each epoch
        
        # set to record loss
        epoch_loss = 0
        epoch_loss_dict = {}
        
        # train for this epoch
        model.train()
        for batch_data in tqdm(train_dataloader, desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(args.device), batch_data))
            loss, loss_dict = self.loss(model, batch_data)
            loss.backward()
            self.optimizer.step()
        
            # record loss
            epoch_loss += loss.item() / len(train_dataloader)
            for loss_name, loss_value in loss_dict.items():
                _loss_val = float(loss_value) / len(train_dataloader)
                if loss_name not in epoch_loss_dict:
                    epoch_loss_dict[loss_name] = _loss_val
                else:
                    epoch_loss_dict[loss_name] += _loss_val

        writer.add_scalar('Loss/train', epoch_loss, epoch_idx)
        self.logger.log_loss(epoch_idx, epoch_loss_dict)

    @log_exceptions
    def train(self, model):
        self.create_optimizer(model)
        now_patience = 0
        best_epoch = 0
        best_metric = -1e9
        for epoch_idx in range(args.epoch):
            # train
            self.train_epoch(model, epoch_idx)
            
            # evaluate
            if epoch_idx % args.tst_step == 0:
                eval_result = self.evaluate(model, epoch_idx)

                # update best weight
                if eval_result[args.metrics[args.criterion[0]]][args.criterion[1]] > best_metric:
                    now_patience = 0
                    best_epoch = epoch_idx
                    best_metric = eval_result[args.metrics[args.criterion[0]]][args.criterion[1]]
                    torch.save(model.state_dict(), f'{args.path}/best.pt')
                    self.logger.log(">>> Validation score increased.  Copying the best model ...")
                else:
                    now_patience += 1

                # early stop
                if (args.patience is not None) and now_patience == args.patience:
                    self.logger.log(f">>> Early stop at epoch {epoch_idx}")
                    break

        # re-initialize the model and load the best parameter
        self.logger.log(">>> Best Epoch {}".format(best_epoch))
        best_model = build_model(self.data_handler).to(args.device)
        best_model.load_state_dict(torch.load(f'{args.path}/best.pt'))
        # self.evaluate(best_model) # activate if best performance in validation is needed
        return best_model

    @log_exceptions
    def evaluate(self, model, epoch_idx=None):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.valid_dataloader)
        if epoch_idx:
            for metric in args.metrics:
                for i, k in enumerate(args.ks):
                    writer.add_scalar(f'{metric}/@{k}', eval_result[metric][i], epoch_idx)
        self.logger.log("[VALID]")
        self.logger.log_eval(eval_result, args.ks, epoch_idx=epoch_idx)
        return eval_result

    @log_exceptions
    def test(self, model):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
        self.logger.log("[TEST]")
        self.logger.log_eval(eval_result, args.ks)
        if args.summary is not None:
            self.logger.log_summary(eval_result, args.ks)
        return eval_result