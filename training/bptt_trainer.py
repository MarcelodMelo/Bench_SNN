from base import BaseTrainer
from snntorch import functional as SF
import torch.optim as optim
class BPTTTrainer(BaseTrainer):
    """Backpropagation Through Time"""
    def __init__(self, model, config):
        super().__init__(model, config)
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        self.loss_fn = SF.ce_rate_loss()
        
    def train_epoch(self, dataloader):
        for batch in dataloader:
            # BPTT automático com loop temporal
            spk_rec = self.forward_pass(batch)
            loss = self.loss_fn(spk_rec, targets)
            loss.backward()  # BPTT através dos timesteps
            self.optimizer.step()