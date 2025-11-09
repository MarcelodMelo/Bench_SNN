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

from .mnist import MNISTDataset
from .bciciv2a import bciciv2A
from .bciciv2b import bciciv2B

__all__ = ['MNISTDataset', 'bciciv2A', 'bciciv2B']