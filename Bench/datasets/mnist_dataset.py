import time
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import format_execution_time

def load_mnist():
    """
    Carrega MNIST no formato tensor padronizado
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])
    
    # ✅ DOWNLOAD SÓ SE NECESSÁRIO (evita download repetido)
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # ✅ CONVERSÃO MAIS EFICIENTE
    X_train = torch.stack([x for x, _ in mnist_train])
    y_train = torch.tensor([y for _, y in mnist_train])
    X_test = torch.stack([x for x, _ in mnist_test])
    y_test = torch.tensor([y for _, y in mnist_test])
    
    return {
        'X_train': X_train,          # [60000, 1, 28, 28]
        'y_train': y_train,          # [60000]
        'X_test': X_test,            # [10000, 1, 28, 28]
        'y_test': y_test,            # [10000]
        'input_shape': (1, 28, 28),
        'num_classes': 10,
        'dataset_name': 'MNIST'
    }

def visualize_mnist_sample(dataset_dict, sample_idx=0, save_path='fig.png'):
    """
    Visualiza amostra do MNIST com valores dos pixels
    """
    images = dataset_dict['X_train']
    labels = dataset_dict['y_train']
    
    image_tensor = images[sample_idx][0]  # [28, 28]
    image_np = image_tensor.numpy()
    label = labels[sample_idx].item()
    
    # ✅ PLOT SIMPLIFICADO
    plt.figure(figsize=(10, 8))
    plt.imshow(image_np, cmap='gray')
    plt.title(f'MNIST - Dígito: {label}', fontsize=14, pad=20)
    
    # ✅ ANOTAÇÕES APENAS PIXELS SIGNIFICATIVOS
    for i in range(28):
        for j in range(28):
            pixel_val = image_np[i, j]
            if pixel_val > 0.01:
                text_color = 'white' if pixel_val < 0.5 else 'black'
                plt.text(j, i, f'{pixel_val:.2f}', ha='center', va='center', 
                        color=text_color, fontsize=6,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor='gray', alpha=0.3))

    plt.axis('off')
    
    # ✅ ESTATÍSTICAS SIMPLIFICADAS
    stats_text = (f"Mín: {image_np.min():.3f} | Máx: {image_np.max():.3f} | "
                  f"Média: {image_np.mean():.3f}\n"
                  f"Pixels ≠ 0: {np.sum(image_np > 0.01)}/784")
    
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Amostra {sample_idx} salva em: {save_path}")
        plt.close()  # Fecha a figura para liberar memória
    else:
        plt.show()


if __name__ == "__main__":
    start_time = time.time()
    print("=== TESTE MNIST ===")
    
    dataset = load_mnist()
    print(f"   Train: {dataset['X_train'].shape} | Test: {dataset['X_test'].shape}")
    print(f"   Input: {dataset['input_shape']} | Classes: {dataset['num_classes']}")
    visualize_mnist_sample(dataset, 0)

    execution_time = time.time() - start_time
    print(f"✅ Execução concluída em {format_execution_time(execution_time)}")
    print("=== TESTE FINALIZADO ===")