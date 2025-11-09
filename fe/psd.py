import torch
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from .base_extractor import BaseFeatureExtractor

class PSDExtractor(BaseFeatureExtractor):
    """
    Extrai Power Spectral Density usando Welch's method
    """
    
    def __init__(self, config=None):
        default_config = {
            'fs': 100,           # Frequência de amostragem
            'nperseg': 256,      # Tamanho do segmento
            'noverlap': 128,     # Overlap entre segmentos
            'freq_bands': {      # Bandas de frequência
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45)
            }
        }
        super().__init__(config or default_config)
        self.feature_names = list(self.config['freq_bands'].keys())
        
    def extract(self, dataset_dict):
        """Extrai PSD do dataset"""
        self._validate_input(dataset_dict)
        print("🔧 Extraindo PSD...")
        
        # Processa dados de treino e teste
        X_train_psd = self._extract_psd_batch(dataset_dict['X_train'])
        X_test_psd = self._extract_psd_batch(dataset_dict['X_test'])
        
        return self._create_output_dict(
            dataset_dict, X_train_psd, X_test_psd, self.feature_names
        )
    
    def _extract_psd_batch(self, data):
        """Extrai PSD para batch de dados"""
        batch_size = data.shape[0]
        num_bands = len(self.config['freq_bands'])
        
        # Inicializa tensor de features
        features = torch.zeros(batch_size, num_bands)
        
        for i in range(batch_size):
            # Converte para numpy e processa cada amostra
            signal_data = data[i].numpy().flatten()
            if len(signal_data) > 1:  # Evita sinais muito curtos
                band_powers = self._compute_band_powers(signal_data)
                features[i] = torch.tensor(band_powers)
        
        return features
    
    def _compute_band_powers(self, signal_data):
        """Calcula potência por banda de frequência"""
        fs = self.config['fs']
        nperseg = min(self.config['nperseg'], len(signal_data))
        
        # Calcula PSD usando Welch
        freqs, psd = signal.welch(
            signal_data, 
            fs=fs, 
            nperseg=nperseg,
            noverlap=self.config['noverlap']
        )
        
        # Calcula potência por banda
        band_powers = []
        for band_name, (low_freq, high_freq) in self.config['freq_bands'].items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                band_powers.append(band_power)
            else:
                band_powers.append(0.0)
        
        return band_powers
    
    def plot_psd_example(self, signal_data, title="PSD Example"):
        """Plota PSD de exemplo"""
        fs = self.config['fs']
        nperseg = min(self.config['nperseg'], len(signal_data))
        
        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg)
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(freqs, psd)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectral Density [V²/Hz]')
        plt.title(title)
        plt.grid(True)
        
        # Destaca bandas de frequência
        for band_name, (low_freq, high_freq) in self.config['freq_bands'].items():
            plt.axvspan(low_freq, high_freq, alpha=0.3, label=band_name)
        
        plt.legend()
        plt.tight_layout()
        plt.show()

def test_psd_extractor():
    """Testa o extrator PSD"""
    import time
    print("=== TESTE PSD EXTRACTOR ===")
    start_time = time.time()
    
    # Dataset simulado (sinais temporais)
    dummy_data = {
        'X_train': torch.randn(50, 1, 1000),  # 50 amostras, 1 canal, 1000 pontos
        'y_train': torch.randint(0, 2, (50,)),
        'X_test': torch.randn(10, 1, 1000),
        'y_test': torch.randint(0, 2, (10,)),
        'input_shape': (1, 1000),
        'num_classes': 2,
        'dataset_name': 'EEG_SIM'
    }
    
    extractor = PSDExtractor({'fs': 100})
    result = extractor.extract(dummy_data)
    
    print(f"📥 Input shape: {dummy_data['X_train'].shape}")
    print(f"📤 Output shape: {result['X_train'].shape}")
    print(f"🎯 Features: {result['feature_names']}")
    print(f"🔢 Número de features: {result['X_train'].shape[1]}")
    
    # Plota exemplo
    sample_signal = dummy_data['X_train'][0, 0, :].numpy()
    extractor.plot_psd_example(sample_signal, "PSD - Sinal de Exemplo")
    
    execution_time = time.time() - start_time
    print(f"✅ PSD concluído em {format_execution_time(execution_time)}")
    print("=== TESTE FINALIZADO ===\n")

if __name__ == "__main__":
    test_psd_extractor()