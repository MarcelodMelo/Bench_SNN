import time
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import format_execution_time
from .base_dataset import BaseDataset # Retira o ponto se o teste for interno

from bciflow.datasets import bciciv2a
from bciflow.modules.tf.bandpass.convolution import bandpass_conv

class bciciv2A(BaseDataset):
    def __init__(self):
        super().__init__()
        eeg_data = bciciv2a(subject=1, path='datasets/data/BCICIV2a/') #Carrego dataset completo
        eeg_data = bandpass_conv(eeg_data, 7, 35)
        eeg_data['X'] = eeg_data['X'][:,:,:,750:1500]
        self.eeg_data = eeg_data
        self.load_raw()
    
    def _load_and_cache_2a(self, eeg_data, remake = True):
        data_path = self._get_data_path()
        cache_file = os.path.join(data_path, "processed/2a.pt")
        
        if os.path.exists(cache_file) and not remake:
            return torch.load(cache_file)
        print("Gerando Cache File")
        X_train, X_test, y_train, y_test = train_test_split(
            eeg_data['X'], 
            eeg_data['y'], 
            test_size=0.2, 
            random_state=42
        )
        # Convertendo para tensores (se não forem)
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(y_test, dtype=torch.long)
        
        cache_data = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test
        }
        torch.save(cache_data, cache_file)
        
        return cache_data
    
    def load_raw(self):
        self.data = self._load_and_cache_2a(self.eeg_data)
        #self._get_train_test({'X': eeg_data['X'], 'y': eeg_data['y']}) #Defino data
        self.metadata = {key: value for key, value in self.eeg_data.items() if key not in ['X', 'y']}
        return self
    
    def plot_eeg_simple(self, sample_idx=0, scale_factor=1000000,save_path='datasets/fig/2a.png'):
        """
        Versão simplificada para visualização rápida.
        """
        # if hasattr(x_train, 'numpy'):
        #     data = x_train.numpy()
        # else:
        #     data = x_train
        data = self.eeg_data['X']
        # Média entre bandas e pegar amostra
        eeg_data = np.mean(data[sample_idx], axis=0)  # [eletrodos, tempo]
        eeg_data_scaled = eeg_data * scale_factor
        plt.figure(figsize=(12, 8))
        
        n_electrodes = eeg_data_scaled.shape[0]
        for electrode in range(n_electrodes):
            offset = electrode * 50
            plt.plot(eeg_data_scaled[electrode] + offset, label=self.eeg_data['ch_names'][electrode])
        
        plt.xlabel('Tempo (amostras)')
        plt.ylabel('Amplitude (μV) + Offset')
        plt.title(f'Sinais de EEG - Amostra {sample_idx} (escala μV)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Amostra {sample_idx} salva em: {save_path}")
            plt.close()
        else:
            plt.show()

        
if __name__ == "__main__":
    start_time = time.time()
    print("==TESTE 2A==")

    dataset = bciciv2A()
    data = dataset.get_data()
    metadata = dataset.get_metadata()
    print(f"Train: {data['X_train'].shape} | Test: {data['X_test'].shape}")
    for key, value in metadata.items():
        print(f'{key}: {value}')

    dataset.plot_eeg_simple(0)
    execution_time = time.time() - start_time
    print(f"Execução concluída em {format_execution_time(execution_time)}")
    print("==TESTE FINALIZADO==\n")