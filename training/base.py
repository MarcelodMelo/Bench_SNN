# training/base_trainer.py
class BaseTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = None
        self.loss_fn = None
        
    def train_epoch(self, dataloader):
        raise NotImplementedError
        
    def validate(self, dataloader):
        raise NotImplementedError
        
    def save_checkpoint(self):
        pass
        
    def load_checkpoint(self):
        pass