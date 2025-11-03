import torch.nn as nn
import snntorch as snn
from base import BaseSNN

# networks/feedforward.py
class FeedForwardSNN(BaseSNN):
    """784i → 256fc → LIF → 10fc → LIF"""
    def __init__(self, config):
        super().__init__(config)
        # Camadas usando config
        self.fc1 = nn.Linear(config.input_size, config.hidden_size)
        self.lif1 = snn.Leaky(beta=config.beta)
        # ...

