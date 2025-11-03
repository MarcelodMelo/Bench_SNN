"""
Datasets Module
---------------
Exporta todos os datasets com interface padronizada

Formato padrão de retorno: dict com:
- 'X_train', 'y_train', 'X_test', 'y_test': tensores
- 'input_shape': tuple sem batch dimension  
- 'num_classes': int
- 'sample_rate': float (para EEG)
- 'channels': list (para EEG)
"""

from .mnist_dataset import load_mnist, visualize_mnist_sample

__all__ = ['load_mnist', 'visualize_mnist_sample']