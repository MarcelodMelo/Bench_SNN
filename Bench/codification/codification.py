import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# =============================================================================
# CONFIGURAÇÃO CENTRALIZADA
# =============================================================================
BSA_CONFIG = {
    'fir_size': 6,        # Tamanho do filtro gaussiano  
    'sigma': 0.8,         # Largura do filtro
    'threshold': 0.15,    # Limiar para gerar spikes
    'min_intensity': 0.05, # Intensidade mínima para processar pixel
    'noise_std': 0.1,     # Desvio do ruído para variação temporal
    'gif_interval': 300,  # Intervalo do GIF (ms)
    'gif_fps': 4,         # Frames por segundo do GIF
}

# =============================================================================
# FUNÇÕES PRINCIPAIS - BSA QUE DITA O NÚMERO DE FRAMES
# =============================================================================

def create_gaussian_filter():
    size = BSA_CONFIG['fir_size']
    sigma = BSA_CONFIG['sigma']
    x = np.linspace(-2, 2, size)
    gaussian = np.exp(-x**2 / (2 * sigma**2))
    return gaussian / np.sum(gaussian)

def bsa_encode_signal(signal, fir_filter):
    """BSA que retorna TODOS os spikes encontrados"""
    L, F = len(signal), len(fir_filter)
    spikes = []
    threshold = BSA_CONFIG['threshold']
    
    s_min, s_max = np.min(signal), np.max(signal)
    if s_max - s_min < 0.001: 
        return np.array([])  # Retorna array vazio se não houver spikes
    
    s_normalized = (signal - s_min) / (s_max - s_min)
    s_copy = s_normalized.copy()
    
    for t in range(L - F):
        err1 = sum(abs(s_copy[t+k] - fir_filter[k]) for k in range(F))
        err2 = sum(abs(s_copy[t+k]) for k in range(F))
        
        if err1 <= (err2 - threshold) and err2 > 0.1:
            spikes.append(t)  # ✅ Armazena o TIMESTEP do spike
            for k in range(F):
                if t + k < L: 
                    s_copy[t+k] = max(0, s_copy[t+k] - fir_filter[k])
    
    return np.array(spikes)

def bsa_encode_batch(images):
    """BSA que gera matriz de spikes com número variável de timesteps"""
    batch_size = images.shape[0]
    images_np = images.numpy().reshape(batch_size, -1)
    min_intensity = BSA_CONFIG['min_intensity']
    noise_std = BSA_CONFIG['noise_std']
    
    fir_filter = create_gaussian_filter()
    
    # ✅ PRIMEIRO: Encontra todos os spikes para determinar o número de frames
    all_spike_timesteps = []
    
    for batch_idx in range(batch_size):
        for pixel_idx in range(images_np.shape[1]):
            intensity = images_np[batch_idx, pixel_idx]
            if intensity > min_intensity:
                # Cria sinal temporal
                temporal_signal = np.ones(50) * intensity  # ✅ Sinal longo (50 timesteps)
                temporal_signal += np.random.normal(0, noise_std, 50)
                temporal_signal = np.clip(temporal_signal, 0, 1)
                
                # Obtém timesteps de spikes
                spike_times = bsa_encode_signal(temporal_signal, fir_filter)
                all_spike_timesteps.extend(spike_times)
    
    # ✅ Número de frames = maior timestep de spike + 1
    if len(all_spike_timesteps) > 0:
        num_frames = int(np.max(all_spike_timesteps)) + 1
    else:
        num_frames = 10  # Fallback
    
    print(f"✅ BSA gerou {len(all_spike_timesteps)} spikes em {num_frames} timesteps")
    
    # ✅ SEGUNDO: Gera matriz de spikes com número correto de frames
    spike_data = np.zeros((num_frames, batch_size, images_np.shape[1]))
    
    for batch_idx in range(batch_size):
        for pixel_idx in range(images_np.shape[1]):
            intensity = images_np[batch_idx, pixel_idx]
            if intensity > min_intensity:
                temporal_signal = np.ones(50) * intensity
                temporal_signal += np.random.normal(0, noise_std, 50)
                temporal_signal = np.clip(temporal_signal, 0, 1)
                
                spike_times = bsa_encode_signal(temporal_signal, fir_filter)
                # Marca spikes nos timesteps corretos
                for t in spike_times:
                    if t < num_frames:  # Só marca se estiver dentro do range
                        spike_data[t, batch_idx, pixel_idx] = 1
    
    return torch.tensor(spike_data, dtype=torch.float32)

def bsa_encode_single(image):
    """BSA para imagem única"""
    if len(image.shape) == 2: 
        image = image.unsqueeze(0)
    return bsa_encode_batch(image.unsqueeze(0))

def visualize_bsa(original_image, spike_data, filename='bsa_encoding.gif'):
    """Gera GIF com número de frames DEFINIDO pelo BSA"""
    num_frames = spike_data.shape[0]  # ✅ Número de frames = número de timesteps com spikes
    original_img_np = original_image[0].numpy() if len(original_image.shape) == 3 else original_image.numpy()
    
    # Diagnóstico
    total_spikes = torch.sum(spike_data).item()
    active_pixels = torch.any(spike_data > 0, dim=0).sum().item()
    print(f"🎯 BSA: {num_frames} frames, {total_spikes} spikes, {active_pixels} pixels ativos")
    
    # Configura animação
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(original_img_np, cmap='gray')
    im2 = ax2.imshow(spike_data[0, 0].view(28, 28).numpy(), cmap='hot', vmin=0, vmax=1)
    ax1.set_title('Original'); ax2.set_title(f'Frame 0/{num_frames}'); 
    ax1.axis('off'); ax2.axis('off')
    
    def animate(frame):
        spike_map = spike_data[frame, 0].view(28, 28).numpy()
        im2.set_array(spike_map)
        frame_spikes = np.sum(spike_map)
        ax2.set_title(f'Frame {frame}/{num_frames}\nSpikes: {frame_spikes:.0f}')
        return [im2]
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                 interval=BSA_CONFIG['gif_interval'], blit=False)
    anim.save(filename, writer='pillow', fps=BSA_CONFIG['gif_fps'])
    plt.close()
    
    print(f"✅ GIF salvo: {num_frames} frames")
    return HTML(anim.to_jshtml())

# =============================================================================
# TESTE
# =============================================================================

def test_bsa():
    """Testa o BSA com número de frames dinâmico"""
    from torchvision import datasets, transforms
    
    # Carrega imagem real
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=True)
    image, label = next(iter(test_loader))
    
    print(f"=== TESTE BSA - Dígito: {label.item()} ===")
    
    # Gera spikes - O NÚMERO DE FRAMES É DEFINIDO AQUI
    spikes = bsa_encode_single(image[0])
    
    # Visualiza
    return visualize_bsa(image[0], spikes)

# Executa teste
if __name__ == "__main__":
    test_bsa()