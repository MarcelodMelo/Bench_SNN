import torch
import torch.nn as nn
import snntorch as snn
from .base_network import BaseSNN

class FeedForwardSNN(BaseSNN):
    """
    FeedForward SNN: 784i-20tBSA-256fc-LIF-10fc-LIF
    Input(784) → Dense(256) → LIF(β=0.9) → Dense(10) → LIF(β=0.9)
    """
    
    def __init__(self, num_inputs=784, num_hidden=256, num_outputs=10, 
                 beta=0.9, spike_grad=None, num_steps=20, config=None):
        super().__init__(config)
        
        self.num_steps = num_steps
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        # Camadas lineares
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        
        # Neurônios LIF
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
    def init_mem(self, batch_size=1):
        """Inicializa estados de memória - CORREÇÃO AQUI"""
        return (
            self.lif1.init_leaky(),  # ✅ CORRIGIDO
            self.lif2.init_leaky()   # ✅ CORRIGIDO
        )
        
    def forward(self, x):
        """
        Forward pass temporal
        Args:
            x: Tensor [batch_size, timesteps, num_inputs]
        Returns:
            spk_rec: Spikes de saída [timesteps, batch_size, num_outputs]
            mem_rec: Memórias de saída [timesteps, batch_size, num_outputs]
        """
        # Inicializar memórias
        mem1, mem2 = self.init_mem()
        
        # Listas para registro
        spk2_rec = []
        mem2_rec = []
        
        # Loop temporal
        for step in range(self.num_steps):
            # Camada 1: Input → Hidden
            cur1 = self.fc1(x[:, step])  # [batch, num_inputs] → [batch, hidden]
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Camada 2: Hidden → Output
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Registrar saídas
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            
        return torch.stack(spk2_rec), torch.stack(mem2_rec)
    
    def __repr__(self):
        return f"FeedForwardSNN({self.num_inputs}→{self.num_hidden}→{self.num_outputs}, timesteps={self.num_steps})"