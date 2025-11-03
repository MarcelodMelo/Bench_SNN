from utils.helpers import format_execution_time
import time
# train_mnist.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import snntorch as snn
from snntorch import surrogate, functional as SF

# Nossos módulos
from networks.feedforward_snn import FeedForwardSNN
from training.bptt_trainer import BPTTTrainer

# Seus módulos
from datasets.mnist_dataset import load_mnist
from encoding.rate_coding import convert_to_rate_coding

def create_spike_dataloader(spikes_dataset, batch_size=128, shuffle=True):
    """Cria DataLoader a partir do dataset de spikes"""
    dataset = TensorDataset(
        spikes_dataset['X_train'], 
        spikes_dataset['y_train']
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    start_time = time.time()
    # Configurações
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    # 1. Carrega e codifica MNIST
    print("Dataset: MNIST")
    original_dataset = load_mnist()
    print(f"Tempo: {format_execution_time(time.time() - start_time)}")

    print("\nEncoding: Rate_Coding")
    spikes_dataset = convert_to_rate_coding(original_dataset, num_steps=20, gain=1.0)
    print(f"Tempo: {format_execution_time(time.time() - start_time)}")
    #print(f"✅ Dados codificados: {spikes_dataset['X_train'].shape}")
    
    # 2. Cria modelo
    model = FeedForwardSNN(
        num_inputs=784,
        num_hidden=256, 
        num_outputs=10,
        beta=0.9,
        spike_grad=surrogate.fast_sigmoid(slope=25),
        num_steps=20
    ).to(device)
    
    print(f"\nModelo: {model}")
    print(f"Parâmetros: {model.get_num_params():,}")
    print(f"Tempo: {format_execution_time(time.time() - start_time)}")
    
    # 3. Otimizador e Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    
    # 4. Trainer
    trainer = BPTTTrainer(model, optimizer, device=device)
    
    # 5. DataLoaders dos spikes
    print(f"\nDataloader")
    train_loader = create_spike_dataloader(spikes_dataset, batch_size=128, shuffle=True)

    test_dataset = TensorDataset(
        spikes_dataset['X_test'],
        spikes_dataset['y_test']
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print(f"Dataloaders: Train {len(train_loader)} batches, Test {len(test_loader)} batches")
    print(f"Tempo: {format_execution_time(time.time() - start_time)}")
    
    # 6. Treinamento
    print("\n" + "="*50)
    print("INICIANDO TREINAMENTO!")
    print("="*50)
    train_losses, train_accs, test_accs = trainer.train(
        train_loader, test_loader, num_epochs=3
    )
    print("="*50)
    print("FINALIZANDO TREINAMENTO!")
    print("="*50)
    
    print(f"\n✅ Treinamento concluído!")
    print(f"🎯 Acurácia final: {test_accs[-1]*100:.2f}%")
    print(f"Tempo: {format_execution_time(time.time() - start_time)}")