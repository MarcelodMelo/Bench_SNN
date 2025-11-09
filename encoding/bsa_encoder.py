import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import sys
import os

# ✅ Importação padrão do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import format_execution_time

# =============================================================================
# CONFIGURAÇÃO CENTRALIZADA
# =============================================================================
BSA_CONFIG = {
    'fir_size': 6,        # Tamanho do filtro gaussiano  
    'sigma': 0.8,         # Largura do filtro
    'threshold': 0.15,    # Limiar para gerar spikes
    'min_intensity': 0.05, # Intensidade mínima para processar pixel
    'noise_std': 0.1,     # Desvio do ruído para variação temporal
}

# =============================================================================
# FUNÇÕES PRINCIPAIS
# =============================================================================

def create_gaussian_filter():
    """Cria filtro gaussiano para BSA"""
    size, sigma = BSA_CONFIG['fir_size'], BSA_CONFIG['sigma']
    x = np.linspace(-2, 2, size)
    gaussian = np.exp(-x**2 / (2 * sigma**2))
    return gaussian / np.sum(gaussian)

def bsa_encode_signal(signal, fir_filter):
    """BSA para sinal 1D - retorna timesteps com spikes"""
    L, F, threshold = len(signal), len(fir_filter), BSA_CONFIG['threshold']
    spikes = []
    
    s_min, s_max = np.min(signal), np.max(signal)
    if s_max - s_min < 0.001: 
        return np.array([])
    
    s_normalized = (signal - s_min) / (s_max - s_min)
    s_copy = s_normalized.copy()
    
    for t in range(L - F):
        err1 = sum(abs(s_copy[t+k] - fir_filter[k]) for k in range(F))
        err2 = sum(abs(s_copy[t+k]) for k in range(F))
        
        if err1 <= (err2 - threshold) and err2 > 0.1:
            spikes.append(t)
            for k in range(F):
                if t + k < L: 
                    s_copy[t+k] = max(0, s_copy[t+k] - fir_filter[k])
    
    return np.array(spikes)

def bsa_encode(dataset_dict):
    """
    Aplica codificação BSA no dataset completo
    
    Args:
        dataset_dict (dict): Dataset no formato padronizado
        
    Returns:
        dict: Spikes no formato padronizado
    """
    print("🔧 Aplicando codificação BSA...")
    
    X_train, X_test = dataset_dict['X_train'], dataset_dict['X_test']
    min_intensity, noise_std = BSA_CONFIG['min_intensity'], BSA_CONFIG['noise_std']
    fir_filter = create_gaussian_filter()
    
    # Processa treino e teste
    spikes_train = _encode_images(X_train, fir_filter, min_intensity, noise_std)
    spikes_test = _encode_images(X_test, fir_filter, min_intensity, noise_std)
    
    # ✅ RETORNO PADRONIZADO
    return {
        'X_train': spikes_train,
        'y_train': dataset_dict['y_train'],
        'X_test': spikes_test, 
        'y_test': dataset_dict['y_test'],
        'input_shape': spikes_train.shape[1:],  # (timesteps, neurons)
        'num_classes': dataset_dict['num_classes'],
        'dataset_name': f"{dataset_dict['dataset_name']}_BSA",
        'encoding_config': BSA_CONFIG,
        'timesteps': spikes_train.shape[1],
    }

def _encode_images(images, fir_filter, min_intensity, noise_std):
    """Codifica batch de imagens para spikes"""
    batch_size = images.shape[0]
    images_flat = images.reshape(batch_size, -1)
    num_pixels = images_flat.shape[1]
    
    # Descobre número de timesteps necessário
    max_timestep = _find_max_timestep(images_flat, fir_filter, min_intensity, noise_std)
    num_timesteps = max(10, max_timestep + 1)  # Mínimo 10 timesteps
    
    # Gera matriz de spikes
    spikes = np.zeros((batch_size, num_timesteps, num_pixels))
    
    for batch_idx in range(batch_size):
        for pixel_idx in range(num_pixels):
            intensity = images_flat[batch_idx, pixel_idx]
            if intensity > min_intensity:
                signal = _create_temporal_signal(intensity, num_timesteps + 20, noise_std)
                spike_times = bsa_encode_signal(signal, fir_filter)
                for t in spike_times:
                    if t < num_timesteps:
                        spikes[batch_idx, t, pixel_idx] = 1
    
    return torch.tensor(spikes, dtype=torch.float32)

def _find_max_timestep(images_flat, fir_filter, min_intensity, noise_std):
    """Encontra o maior timestep com spikes"""
    max_time = 0
    for batch_idx in range(min(10, images_flat.shape[0])):  # Amostra para eficiência
        for pixel_idx in range(min(100, images_flat.shape[1])):  # Amostra pixels
            intensity = images_flat[batch_idx, pixel_idx]
            if intensity > min_intensity:
                signal = _create_temporal_signal(intensity, 50, noise_std)
                spike_times = bsa_encode_signal(signal, fir_filter)
                if len(spike_times) > 0:
                    max_time = max(max_time, np.max(spike_times))
    return max_time

def _create_temporal_signal(intensity, length, noise_std):
    """Cria sinal temporal a partir da intensidade"""
    intensity_val = intensity.item() if torch.is_tensor(intensity) else intensity
    signal = np.ones(length) * intensity_val
    signal += np.random.normal(0, noise_std, length)
    return np.clip(signal, 0, 1)

def visualize_bsa_encoding(spike_data, original_data=None, sample_idx=0, save_path=None):
    """
    Visualiza codificação BSA de uma amostra
    
    Args:
        spike_data (dict): Spikes no formato padronizado
        original_data (dict): Dataset original (opcional)
        sample_idx (int): Índice da amostra
        save_path (str): Caminho para salvar imagem
    """
    spikes = spike_data['X_train'][sample_idx]  # [timesteps, neurons]
    label = spike_data['y_train'][sample_idx].item()
    timesteps = spike_data['timesteps']
    
    # Prepara figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) if original_data else plt.subplots(1, 1, figsize=(6, 5))
    
    if original_data:
        # Mostra imagem original
        original_img = original_data['X_train'][sample_idx][0].numpy()
        axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title(f'Original - Dígito: {label}')
        axes[0].axis('off')
        ax_spikes = axes[1]
    else:
        ax_spikes = axes
    
    # Mostra spikes acumulados
    spike_count = torch.sum(spikes, dim=0).view(28, 28).numpy()
    im = ax_spikes.imshow(spike_count, cmap='hot')
    ax_spikes.set_title(f'BSA - Spikes acumulados\nTotal: {torch.sum(spikes).item():.0f}')
    ax_spikes.axis('off')
    plt.colorbar(im, ax=ax_spikes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Visualização salva: {save_path}")
        plt.close()
    else:
        plt.show()

# =============================================================================
# TESTE
# =============================================================================

def test_bsa_encoder():
    """Testa o encoder BSA completo"""
    import time
    print("=== TESTE BSA ENCODER ===")
    start_time = time.time()
    
    # Carrega dados
    from datasets.mnist import load_mnist
    dataset = load_mnist()
    
    print(f"📥 Input: {dataset['X_train'].shape}")
    
    # Aplica BSA
    spikes_data = bsa_encode(dataset)
    
    print(f"📤 Output: {spikes_data['X_train'].shape}")
    print(f"🎯 Timesteps: {spikes_data['timesteps']}")
    print(f"🔢 Spikes totais: {torch.sum(spikes_data['X_train']).item():.0f}")
    
    # Visualiza
    visualize_bsa_encoding(spikes_data, dataset, save_path='bsa_sample.png')
    
    execution_time = time.time() - start_time
    print(f"✅ BSA concluído em {format_execution_time(execution_time)}")

if __name__ == "__main__":
    test_bsa_encoder()