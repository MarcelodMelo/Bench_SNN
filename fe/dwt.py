import torch
import numpy as np
import pywt
import matplotlib.pyplot as plt
from .base_extractor import BaseFeatureExtractor

class DWTExtractor(BaseFeatureExtractor):
    """
    Extrai características usando Discrete Wavelet Transform
    """
    
    def __init__(self, config=None):
        default_config = {
            'wavelet': 'db4',           # Wavelet a ser usada
            'level': 4,                 # Número de níveis de decomposição
            'features': ['energy', 'std', 'entropy']  # Características a extrair
        }
        super().__init__(config or default_config)
        
    def extract(self, dataset_dict):
        """Extrai características DWT do dataset"""
        self._validate_input(dataset_dict)
        print("🔧 Extraindo DWT...")
        
        # Processa dados de treino e teste
        X_train_dwt = self._extract_dwt_batch(dataset_dict['X_train'])
        X_test_dwt = self._extract_dwt_batch(dataset_dict['X_test'])
        
        feature_names = self._generate_feature_names()
        
        return self._create_output_dict(
            dataset_dict, X_train_dwt, X_test_dwt, feature_names
        )
    
    def _extract_dwt_batch(self, data):
        """Extrai DWT para batch de dados"""
        batch_size = data.shape[0]
        
        features_list = []
        for i in range(batch_size):
            signal_data = data[i].numpy().flatten()
            signal_features = self._extract_signal_features(signal_data)
            features_list.append(signal_features)
        
        return torch.tensor(features_list)
    
    def _extract_signal_features(self, signal_data):
        """Extrai características de um sinal usando DWT"""
        # Decomposição wavelet
        coeffs = pywt.wavedec(signal_data, self.config['wavelet'], level=self.config['level'])
        
        features = []
        for i, coeff in enumerate(coeffs):
            for feat_type in self.config['features']:
                if feat_type == 'energy':
                    features.append(np.sum(coeff**2))
                elif feat_type == 'std':
                    features.append(np.std(coeff))
                elif feat_type == 'entropy':
                    features.append(self._shannon_entropy(coeff))
                elif feat_type == 'mean':
                    features.append(np.mean(coeff))
        
        return features
    
    def _shannon_entropy(self, signal):
        """Calcula entropia de Shannon"""
        if len(signal) == 0:
            return 0
        # Normaliza para distribuição de probabilidade
        prob = np.abs(signal) / np.sum(np.abs(signal))
        prob = prob[prob > 0]  # Remove zeros para log
        return -np.sum(prob * np.log2(prob))
    
    def _generate_feature_names(self):
        """Gera nomes das features baseado na configuração"""
        names = []
        coeff_names = ['A'] + [f'D{i}' for i in range(self.config['level'], 0, -1)]
        
        for coeff_name in coeff_names:
            for feat_type in self.config['features']:
                names.append(f"{coeff_name}_{feat_type}")
        
        return names
    
    def plot_dwt_decomposition(self, signal_data, title="DWT Decomposition"):
        """Plota decomposição DWT de exemplo"""
        coeffs = pywt.wavedec(signal_data, self.config['wavelet'], level=self.config['level'])
        
        fig, axes = plt.subplots(len(coeffs), 1, figsize=(12, 8))
        if len(coeffs) == 1:
            axes = [axes]
        
        coeff_names = ['Approximation'] + [f'Detail {i}' for i in range(len(coeffs)-1, 0, -1)]
        
        for i, (coeff, name) in enumerate(zip(coeffs, coeff_names)):
            axes[i].plot(coeff)
            axes[i].set_title(f'{name} (Length: {len(coeff)})')
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()

def test_dwt_extractor():
    """Testa o extrator DWT"""
    import time
    print("=== TESTE DWT EXTRACTOR ===")
    start_time = time.time()
    
    # Dataset simulado
    dummy_data = {
        'X_train': torch.randn(30, 1, 512),  # 30 amostras, 1 canal, 512 pontos
        'y_train': torch.randint(0, 2, (30,)),
        'X_test': torch.randn(10, 1, 512),
        'y_test': torch.randint(0, 2, (10,)),
        'input_shape': (1, 512),
        'num_classes': 2,
        'dataset_name': 'EEG_SIM'
    }
    
    extractor = DWTExtractor({
        'wavelet': 'db4',
        'level': 4,
        'features': ['energy', 'std', 'entropy']
    })
    
    result = extractor.extract(dummy_data)
    
    print(f"📥 Input shape: {dummy_data['X_train'].shape}")
    print(f"📤 Output shape: {result['X_train'].shape}")
    print(f"🎯 Número de features: {len(result['feature_names'])}")
    print(f"🔤 Primeiras features: {result['feature_names'][:5]}...")
    
    # Plota decomposição de exemplo
    sample_signal = dummy_data['X_train'][0, 0, :].numpy()
    extractor.plot_dwt_decomposition(sample_signal, "DWT - Sinal de Exemplo")
    
    execution_time = time.time() - start_time
    print(f"✅ DWT concluído em {format_execution_time(execution_time)}")
    print("=== TESTE FINALIZADO ===\n")

if __name__ == "__main__":
    test_dwt_extractor()