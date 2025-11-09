import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import format_execution_time

class BaseFeatureExtractor:
    """
    Classe base para todos os extratores de características
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.feature_names = []
        
    def extract(self, dataset_dict):
        """
        Extrai características do dataset
        
        Args:
            dataset_dict (dict): Dataset no formato padronizado
            
        Returns:
            dict: Dataset com características extraídas
        """
        raise NotImplementedError("Método extract deve ser implementado")
    
    def _validate_input(self, dataset_dict):
        """Valida formato de entrada"""
        required_keys = ['X_train', 'y_train', 'X_test', 'y_test', 'input_shape']
        for key in required_keys:
            if key not in dataset_dict:
                raise ValueError(f"Dataset deve conter '{key}'")
    
    def _create_output_dict(self, original_dict, X_train_feat, X_test_feat, feature_names):
        """Cria dicionário de saída padronizado"""
        return {
            'X_train': X_train_feat,
            'y_train': original_dict['y_train'],
            'X_test': X_test_feat,
            'y_test': original_dict['y_test'],
            'input_shape': X_train_feat.shape[1:],
            'num_classes': original_dict['num_classes'],
            'dataset_name': f"{original_dict['dataset_name']}_{self.__class__.__name__}",
            'feature_names': feature_names,
            'extractor_config': self.config
        }

def test_base_extractor():
    """Testa a classe base"""
    print("=== TESTE BASE EXTRACTOR ===")
    
    # Dataset simulado
    dummy_data = {
        'X_train': torch.randn(100, 1, 28, 28),
        'y_train': torch.randint(0, 10, (100,)),
        'X_test': torch.randn(20, 1, 28, 28),
        'y_test': torch.randint(0, 10, (20,)),
        'input_shape': (1, 28, 28),
        'num_classes': 10,
        'dataset_name': 'TEST'
    }
    
    extractor = BaseFeatureExtractor()
    try:
        extractor.extract(dummy_data)
        print("❌ Deveria ter levantado NotImplementedError")
    except NotImplementedError:
        print("✅ Base extractor funcionando corretamente")
    
    print("=== TESTE FINALIZADO ===\n")

if __name__ == "__main__":
    test_base_extractor()