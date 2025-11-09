import time
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import format_execution_time
from .base_dataset import BaseDataset

class MNISTDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.load_raw()
    
    def _load_and_cache_mnist(self):
        data_path = self._get_data_path()
        cache_file = os.path.join(data_path, "processed/mnist.pt")
        
        if os.path.exists(cache_file):
            return torch.load(cache_file)
        
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0,), (1,))
        ])
        
        mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        
        X_train = torch.stack([x for x, _ in mnist_train])
        y_train = torch.tensor([y for _, y in mnist_train])
        X_test = torch.stack([x for x, _ in mnist_test])
        y_test = torch.tensor([y for _, y in mnist_test])
        
        cache_data = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test
        }
        torch.save(cache_data, cache_file)
        
        return cache_data
    
    def load_raw(self):
        cache_data = self._load_and_cache_mnist()
        
        self.data = cache_data
        self.metadata = {
            'input_shape': (1, 28, 28),
            'num_classes': 10,
            'dataset_name': 'MNIST'
        }
        return self

def visualize_mnist_sample(dataset, sample_idx=0, save_path='datasets/fig/mnist.png'):
    data = dataset.get_data()
    images = data['X_train']
    labels = data['y_train']
    
    image_tensor = images[sample_idx][0]
    image_np = image_tensor.numpy()
    label = labels[sample_idx].item()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image_np, cmap='gray')
    plt.title(f'MNIST - Dígito: {label}', fontsize=14, pad=20)
    
    for i in range(28):
        for j in range(28):
            pixel_val = image_np[i, j]
            if pixel_val > 0.01:
                text_color = 'white' if pixel_val < 0.5 else 'black'
                plt.text(j, i, f'{pixel_val:.2f}', ha='center', va='center', 
                        color=text_color, fontsize=6,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor='gray', alpha=0.3))

    plt.axis('off')
    
    stats_text = (f"Mín: {image_np.min():.3f} | Máx: {image_np.max():.3f} | "
                  f"Média: {image_np.mean():.3f}\n"
                  f"Pixels ≠ 0: {np.sum(image_np > 0.01)}/784")
    
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Amostra {sample_idx} salva em: {save_path}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    start_time = time.time()
    print("==TESTE MNIST==")
    
    dataset = MNISTDataset()
    data = dataset.get_data()
    metadata = dataset.get_metadata()
    print(f"Train: {data['X_train'].shape} | Test: {data['X_test'].shape}")
    for key, value in metadata.items():
        print(f'{key}: {value}')


    #visualize_mnist_sample(dataset, 0)
    print(data['X_train'][0])
    execution_time = time.time() - start_time
    print(f"Execução concluída em {format_execution_time(execution_time)}")
    print("==TESTE FINALIZADO==\n")