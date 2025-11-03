import torch.nn as nn
import snntorch as snn

class BaseSNN(nn.Module):
    """Interface base para todas as redes SNN"""
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
    def init_mem(self, batch_size=1):
        """Inicializa estados de memória - deve ser implementado pelas subclasses"""
        raise NotImplementedError
        
    def forward(self, x):
        """Forward pass - deve ser implementado pelas subclasses"""
        raise NotImplementedError
        
    def get_num_params(self):
        """Retorna número total de parâmetros"""
        return sum(p.numel() for p in self.parameters())