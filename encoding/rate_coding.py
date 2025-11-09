# ✅ Importação padrão do projeto
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import format_execution_time
from datasets import *

import torch
from snntorch import spikegen
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def convert_to_rate_coding(data_dict, num_steps=20, gain=1.0):
    """
    Converte dataset para formato de spikes usando rate coding
    
    Args:
        dataset_dict: Dataset no formato original
        num_steps: Número de passos temporais
        gain: Ganho para a codificação
        
    Returns:
        Dataset no formato de spikes
    """
    # Configuração da codificação
    encoding_config = {
        'num_steps': num_steps,
        'gain': gain,
        'encoding_type': 'rate_coding'
    }
    
    # Aplica rate coding aos dados de treino e teste
    X_train_spikes = spikegen.rate(
        data_dict['X_train'].flatten(1),  # [n_amostras, 784]
        num_steps=num_steps,
        gain=gain
    ).permute(1, 0, 2)  # [n_amostras, timesteps, neurônios]
    
    X_test_spikes = spikegen.rate(
        data_dict['X_test'].flatten(1),   # [n_amostras, 784]
        num_steps=num_steps,
        gain=gain
    ).permute(1, 0, 2)  # [n_amostras, timesteps, neurônios]
    
    # Retorna no formato padronizado de spikes
    data = {
        'X_train': X_train_spikes,
        'y_train': data_dict['y_train'],
        'X_test': X_test_spikes,
        'y_test': data_dict['y_test'],
        'input_shape': (num_steps, X_train_spikes.shape[-1]),
    }
    return data
    # return {
    #     'X_train': X_train_spikes,
    #     'y_train': data_dict['y_train'],
    #     'X_test': X_test_spikes,
    #     'y_test': data_dict['y_test'],
    #     'input_shape': (num_steps, X_train_spikes.shape[-1]),  # (timesteps, neurônios)
    #     'num_classes': dataset_dict['num_classes'],
    #     'timesteps': num_steps,
    #     'encoding_config': encoding_config,
    #     'dataset_name': f"{dataset_dict.get('dataset_name', 'dataset')}_rate_coding"
    # }

def create_spike_gif(spikes_dataset, original_dataset, sample_idx=0, gif_path='spikes_evolution.gif'):
    """
    Cria GIF mostrando imagem original + evolução temporal dos spikes
    
    Args:
        spikes_dataset: Dataset no formato de spikes
        original_dataset: Dataset original
        sample_idx: Índice da amostra a visualizar
        gif_path: Caminho para salvar o GIF
    """
    # Pega os dados
    spikes = spikes_dataset['X_train'][sample_idx]  # [timesteps, 784]
    original_img = original_dataset['X_train'][sample_idx][0]  # [28, 28]
    label = original_dataset['y_train'][sample_idx].item()
    timesteps = spikes_dataset['timesteps']
    
    # Configura figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Imagem original (estática)
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title(f'Original - Dígito: {label}')
    ax1.axis('off')
    
    # Spikes (animação)
    spike_map = spikes[0].view(28, 28).numpy()
    im = ax2.imshow(spike_map, cmap='hot', vmin=0, vmax=1)
    ax2.set_title(f'Timestep: 0/{timesteps-1}\nSpikes: {torch.sum(spikes[0]).item()}')
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Função de animação
    def animate(frame):
        spike_map = spikes[frame].view(28, 28).numpy()
        im.set_array(spike_map)
        current_spikes = torch.sum(spikes[frame]).item()
        ax2.set_title(f'Timestep: {frame}/{timesteps-1}\nSpikes: {current_spikes}')
        return [im]
    
    # Cria e salva animação
    anim = animation.FuncAnimation(fig, animate, frames=timesteps, interval=400, blit=False)
    anim.save(gif_path, writer='pillow', fps=3, dpi=100)
    plt.close()
    
    print(f"💾 GIF salvo: {gif_path}")
    return gif_path

if __name__ == "__main__":
    start_time = time.time()
    print("=== TESTE RATE CODING ===")

    dataset = MNISTDataset()
    data = dataset.get_data()
    spikes_dataset = convert_to_rate_coding(data, num_steps=20, gain=1.0)

    print(f"Original: {data['X_train'].shape}")      # [60000, 1, 28, 28]
    print(f"Spikes: {spikes_dataset['X_train'].shape}") # [60000, 20, 784]
    print(f"Novo input_shape: {spikes_dataset['input_shape']}")  # (20, 784)

    # Gera GIF da primeira amostra
    #gif_path = create_spike_gif(spikes_dataset, data, sample_idx=0)
    print(spikes_dataset['X_train'][0])
    execution_time = time.time() - start_time
    print(f"Execução concluída em {format_execution_time(execution_time)}")
    print("=== TESTE FINALIZADO ===")