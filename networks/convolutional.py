import torch
import torch.nn as nn
import snntorch as snn
from .base_network import BaseSNN

class Conv1D_SNN(BaseSNN):
    """
    SNN Convolucional 1D para EEG baseada no paper:
    Input: 22x751 (canais x tempo)
    C1: Feature Maps 3@22x750
    S2: Feature Maps 3@22x150 (Average Pooling)
    C3: Feature Maps 8@22x150  
    S4: Feature Maps 8@22x30 (Average Pooling)
    F5: fc 8x22x30
    Output: 4 classes
    """
    
    def __init__(self, num_input_channels=22, num_timesteps=751, num_classes=4,
                 beta=0.9, spike_grad=None, num_steps=20, dropout_rate=0.5, config=None):
        super().__init__(config)
        
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Camada C1: Conv1D - 3 feature maps @ 22x750
        self.conv1 = nn.Conv1d(
            in_channels=num_input_channels, 
            out_channels=3, 
            kernel_size=2,  # 751 → 750
            stride=1,
            padding=0
        )
        
        # Camada S2: Average Pooling - 3@22x150
        self.pool1 = nn.AvgPool1d(
            kernel_size=5,  # 750 → 150
            stride=5
        )
        
        # Camada C3: Conv1D - 8 feature maps @ 22x150
        self.conv2 = nn.Conv1d(
            in_channels=3, 
            out_channels=8, 
            kernel_size=1,  # Mantém 150
            stride=1,
            padding=0
        )
        
        # Camada S4: Average Pooling - 8@22x30
        self.pool2 = nn.AvgPool1d(
            kernel_size=5,  # 150 → 30
            stride=5
        )
        
        # Calcula o tamanho após as convoluções e pooling
        # 22 canais * 8 feature maps * 30 timesteps = 5280
        self.fc_input_size = 8 * num_input_channels * 30
        
        # Camada F5: Fully Connected
        self.fc1 = nn.Linear(self.fc_input_size, 256)  # Dimensão reduzida
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout para prevenir overfitting
        self.dropout = nn.Dropout(dropout_rate)
        
        # Neurônios LIF
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
    def init_mem(self, batch_size=1):
        """Inicializa estados de memória para todas as camadas LIF"""
        return (
            self.lif1.init_leaky(),  # Após conv1
            self.lif2.init_leaky(),  # Após pool1  
            self.lif3.init_leaky(),  # Após conv2
            self.lif4.init_leaky()   # Após fc1
        )
        
    # No arquivo convolutional.py, modifique o forward:

    def forward(self, x):
        """
        Forward pass temporal
        Args:
            x: Tensor [batch_size, timesteps, num_channels * num_timesteps]
            ou [batch_size, timesteps, num_channels, num_timesteps]
        """
        batch_size, num_steps, *rest = x.shape
        
        # Detectar formato e reorganizar se necessário
        if len(rest) == 1:
            # Formato achatado: [batch, timesteps, channels*timesteps]
            x = x.view(batch_size, num_steps, 22, 751)
        elif len(rest) == 2:
            # Já está no formato correto: [batch, timesteps, channels, timesteps]
            pass
        else:
            raise ValueError(f"Formato de entrada inválido: {x.shape}")
        
        # Resto do código permanece igual...
        mem1, mem2, mem3, mem4 = self.init_mem(batch_size=batch_size)
        
        # Listas para registro
        spk_rec = []
        mem_rec = []
        
        # Loop temporal
        for step in range(self.num_steps):
            # Get current timestep: [batch, channels, timesteps_length]
            x_step = x[:, step]  # [batch, 22, 751]
            
            # === CAMADA C1: Conv1D ===
            c1 = self.conv1(x_step)
            spk1, mem1 = self.lif1(c1, mem1)
            
            # === CAMADA S2: Average Pooling ===  
            s2 = self.pool1(spk1)
            spk2, mem2 = self.lif2(s2, mem2)
            
            # === CAMADA C3: Conv1D ===
            c3 = self.conv2(spk2)
            spk3, mem3 = self.lif3(c3, mem3)
            
            # === CAMADA S4: Average Pooling ===
            s4 = self.pool2(spk3)
            spk4, mem4 = self.lif4(s4, mem4)
            
            # === CAMADA F5: Fully Connected ===
            # Flatten: [batch, 8, 22, 30] → [batch, 8*22*30]
            flattened = spk4.contiguous().view(spk4.size(0), -1)
            
            # Primeira camada fully connected
            fc1_out = self.fc1(flattened)
            fc1_out = self.dropout(fc1_out)
            
            # Camada de saída
            output = self.fc2(fc1_out)
            
            # Registrar saídas
            spk_rec.append(output)
            mem_rec.append(output)
            
        return torch.stack(spk_rec), torch.stack(mem_rec)
    
    def __repr__(self):
        return (f"Conv1D_SNN(22x751→3@22x750→3@22x150→8@22x150→8@22x30→{self.fc_input_size}fc→{self.num_classes}, "
                f"timesteps={self.num_steps}, dropout={self.dropout_rate})")