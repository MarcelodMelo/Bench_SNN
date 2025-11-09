# datasets/base_loader.py
import os
from torch.utils.data import Dataset, TensorDataset, random_split
import torch
from sklearn.model_selection import train_test_split

class BaseDataset:
    def __init__(self):
        self.data = {}       
        self.metadata = {}  
    
    def _get_data_path(self):
        current_dir = os.path.dirname(__file__)
        return os.path.join(current_dir, 'data')
    
    # def _get_train_test(self, dic):
    #     # Separando em treino e teste (80/20)
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         dic['X'], 
    #         dic['y'], 
    #         test_size=0.2, 
    #         random_state=42
    #     )
    #     # Convertendo para tensores (se não forem)
    #     if not isinstance(X_train, torch.Tensor):
    #         X_train = torch.tensor(X_train, dtype=torch.float32)
    #     if not isinstance(y_train, torch.Tensor):
    #         y_train = torch.tensor(y_train, dtype=torch.long)
    #     if not isinstance(X_test, torch.Tensor):
    #         X_test = torch.tensor(X_test, dtype=torch.float32)
    #     if not isinstance(y_test, torch.Tensor):
    #         y_test = torch.tensor(y_test, dtype=torch.long)

    #     self.data = {
    #         'X_train': X_train,
    #         'y_train': y_train, 
    #         'X_test': X_test,
    #         'y_test': y_test
    #     }
    #     return self
            
    
    def load_raw(self):
        # Implementado por cada dataset
        pass
    
    def get_data(self):
        return self.data  
    
    def get_metadata(self):
        return self.metadata.copy() 
    