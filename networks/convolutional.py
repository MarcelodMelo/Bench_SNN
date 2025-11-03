import torch.nn as nn
import snntorch as snn
from networks.base_network import BaseSNN

# networks/convolutional.py  
class ConvSNN(BaseSNN):
    """12C5-MP2-64C5-MP2-1024FC10"""
    def __init__(self, config):
        super().__init__(config)
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=config.beta)
        # ...