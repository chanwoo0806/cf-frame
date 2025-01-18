from cf_frame.configurator import args
from cf_frame.dataloader import DataHandler
from cf_frame.util import init_seed, Logger, build_model, build_trainer, build_loss

def main():
    init_seed()
    logger = Logger()
    
    # Set Loss, DataHandler, Model, and Trainer
    loss = build_loss()
    data_handler = DataHandler(loss.type)
    data_handler.load_data()
    model = build_model(data_handler).to(args.device)
    trainer = build_trainer(data_handler, logger, loss)

    # Train and test the model
    best_model = trainer.train(model)
    trainer.test(best_model)

if __name__ == '__main__':
    main()