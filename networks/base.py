import torch.nn as nn
# networks/base_network.py
class BaseSNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, x):
        raise NotImplementedError
        
    def reset_states(self):
        """Reset estados dos neurônios (importante para SNNs)"""
        pass
        
    def get_spike_counts(self):
        """Retorna estatísticas de spikes"""
        pass