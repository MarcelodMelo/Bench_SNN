import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from .base_extractor import BaseFeatureExtractor

class DispersionEntropyExtractor(BaseFeatureExtractor):
    """
    Extrai Dispersion Entropy para análise de complexidade de sinais
    """
    
    def __init__(self, config=None):
        default_config = {
            'm': 3,          # Embedding dimension
            'c': 6,          # Number of classes
            'tau': 1,        # Time delay
            'type': 'DE'     # 'DE' or 'FDE' (Fuzzy Dispersion Entropy)
        }
        super().__init__(config or default_config)
        
    def extract(self, dataset_dict):
        """Extrai Dispersion Entropy do dataset"""
        self._validate_input(dataset_dict)
        print("🔧 Extraindo Dispersion Entropy...")
        
        # Processa dados de treino e teste
        X_train_de = self._extract_de_batch(dataset_dict['X_train'])
        X_test_de = self._extract_de_batch(dataset_dict['X_test'])
        
        feature_names = ['dispersion_entropy']
        
        return self._create_output_dict(
            dataset_dict, X_train_de, X_test_de, feature_names
        )
    
    def _extract_de_batch(self, data):
        """Extrai DE para batch de dados"""
        batch_size = data.shape[0]
        
        features = torch.zeros(batch_size, 1)
        for i in range(batch_size):
            signal_data = data[i].numpy().flatten()
            de_value = self._dispersion_entropy(signal_data)
            features[i, 0] = de_value
        
        return features
    
    def _dispersion_entropy(self, signal_data):
        """Calcula Dispersion Entropy"""
        m = self.config['m']
        c = self.config['c']
        tau = self.config['tau']
        
        if len(signal_data) < m:
            return 0.0
        
        # Normaliza o sinal
        signal_normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
        
        # Mapeia para classes usando função de distribuição normal
        y = stats.norm.cdf(signal_normalized)
        z = np.round(c * y + 0.5)
        z = np.clip(z, 1, c)  # Garante que está entre 1 e c
        
        # Gera padrões de dispersão
        patterns = []
        for i in range(len(z) - (m - 1) * tau):
            pattern = tuple(z[i + j * tau] for j in range(m))
            patterns.append(pattern)
        
        # Calcula frequência dos padrões
        unique_patterns, counts = np.unique(patterns, return_counts=True, axis=0)
        probabilities = counts / len(patterns)
        
        # Calcula entropia
        entropy = -np.sum(probabilities * np.log(probabilities))
        
        return entropy
    
    def _fuzzy_dispersion_entropy(self, signal_data):
        """Calcula Fuzzy Dispersion Entropy (opcional)"""
        # Implementação simplificada do FDE
        m = self.config['m']
        c = self.config['c']
        
        if len(signal_data) < m:
            return 0.0
        
        # Similar ao DE mas com lógica fuzzy
        signal_normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
        
        # Implementação básica - pode ser expandida
        de_value = self._dispersion_entropy(signal_data)
        return de_value  # Placeholder
    
    def plot_entropy_analysis(self, signal_data, title="Dispersion Entropy Analysis"):
        """Plota análise de entropia"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Sinal original
        axes[0].plot(signal_data)
        axes[0].set_title('Sinal Original')
        axes[0].grid(True)
        
        # Análise de entropia para diferentes parâmetros
        m_values = [2, 3, 4]
        entropies = []
        
        for m in m_values:
            self.config['m'] = m
            entropy = self._dispersion_entropy(signal_data)
            entropies.append(entropy)
        
        axes[1].bar(m_values, entropies)
        axes[1].set_xlabel('Dimensão de Embedding (m)')
        axes[1].set_ylabel('Dispersion Entropy')
        axes[1].set_title('Entropia vs Dimensão')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()

def test_dispersion_entropy():
    """Testa o extrator de Dispersion Entropy"""
    import time
    print("=== TESTE DISPERSION ENTROPY ===")
    start_time = time.time()
    
    # Gera sinais com diferentes complexidades
    np.random.seed(42)
    simple_signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
    complex_signal = np.cumsum(np.random.randn(1000))
    
    # Dataset simulado
    dummy_data = {
        'X_train': torch.stack([
            torch.tensor(simple_signal).unsqueeze(0),
            torch.tensor(complex_signal).unsqueeze(0)
        ]),
        'y_train': torch.tensor([0, 1]),
        'X_test': torch.stack([
            torch.tensor(simple_signal[:500]).unsqueeze(0),
            torch.tensor(complex_signal[:500]).unsqueeze(0)
        ]),
        'y_test': torch.tensor([0, 1]),
        'input_shape': (1, 1000),
        'num_classes': 2,
        'dataset_name': 'ENTROPY_TEST'
    }
    
    extractor = DispersionEntropyExtractor({'m': 3, 'c': 6})
    result = extractor.extract(dummy_data)
    
    print(f"📥 Input shape: {dummy_data['X_train'].shape}")
    print(f"📤 Output shape: {result['X_train'].shape}")
    print(f"🎯 Entropia sinal simples: {result['X_train'][0, 0]:.4f}")
    print(f"🎯 Entropia sinal complexo: {result['X_train'][1, 0]:.4f}")
    
    # Plota análise
    extractor.plot_entropy_analysis(complex_signal, "Análise de Entropia - Sinal Complexo")
    
    execution_time = time.time() - start_time
    print(f"✅ Dispersion Entropy concluído em {format_execution_time(execution_time)}")
    print("=== TESTE FINALIZADO ===\n")

if __name__ == "__main__":
    test_dispersion_entropy()