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
from networks.convolutional import Conv1D_SNN
from training.bptt_trainer import BPTTTrainer

# Seus módulos
from datasets import *
from encoding.rate_coding import convert_to_rate_coding

def standardize_data(data_dict):
    """Padroniza os dados com média 0 e std 1, depois escala para [0, 1]"""
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    
    # Calcula média e std apenas nos dados de treino
    mean = X_train.mean()
    std = X_train.std()
    
    # Padroniza
    X_train_std = (X_train - mean) / (std + 1e-8)
    X_test_std = (X_test - mean) / (std + 1e-8)
    
    # Escala para [0, 1] usando min-max após padronização
    min_val = X_train_std.min()
    max_val = X_train_std.max()
    
    X_train_norm = (X_train_std - min_val) / (max_val - min_val + 1e-8)
    X_test_norm = (X_test_std - min_val) / (max_val - min_val + 1e-8)
    
    data_dict['X_train'] = X_train_norm
    data_dict['X_test'] = X_test_norm
    
    print(f"Standardization: mean={mean:.4f}, std={std:.4f}")
    print(f"Final range: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
    return data_dict

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
    #print("Dataset: MNIST")
    #dataset = MNISTDataset()

    print("Dataset: 2A")
    dataset = bciciv2A()

    #print("Dataset: 2B")
    #dataset = bciciv2B()

    original_dataset = dataset.get_data()
    original_dataset = standardize_data(original_dataset)
    print(f"Tempo: {format_execution_time(time.time() - start_time)}")

    print("\nEncoding: Rate_Coding")
    spikes_dataset = convert_to_rate_coding(original_dataset, num_steps=20, gain=1.0)
    print(f"✅ Dados codificados: {spikes_dataset['X_train'].shape}")
    print(f"Tempo: {format_execution_time(time.time() - start_time)}")
    
    # 2. Cria modelo
    bciciv2a_model = FeedForwardSNN(
        num_inputs=16500,
        num_hidden=256, 
        num_outputs=4,
        beta=0.9,
        spike_grad=surrogate.fast_sigmoid(slope=25),
        num_steps=20
    ).to(device)

    bciciv2b_model = FeedForwardSNN(
        num_inputs=2250,
        num_hidden=256, 
        num_outputs=2,
        beta=0.9,
        spike_grad=surrogate.fast_sigmoid(slope=25),
        num_steps=20
    ).to(device)

    mnist_model = FeedForwardSNN(
        num_inputs=784,
        num_hidden=256, 
        num_outputs=10,
        beta=0.9,
        spike_grad=surrogate.fast_sigmoid(slope=25),
        num_steps=20
    ).to(device)

    # test = Conv1D_SNN(
    #     num_input_channels=22,
    #     num_timesteps=751, 
    #     num_classes=4,
    #     beta=0.9,
    #     spike_grad=surrogate.fast_sigmoid(slope=25),
    #     num_steps=20,
    #     dropout_rate=0.5
    # ).to(device)

    model = bciciv2a_model
    
    
    print(f"\nModelo: {model}")
    print(f"Parâmetros: {model.get_num_params():,}")
    print(f"Tempo: {format_execution_time(time.time() - start_time)}")
    
    # 3. Otimizador e Loss
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
    
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
        train_loader, test_loader, num_epochs=5
    )
    print("="*50)
    print("FINALIZANDO TREINAMENTO!")
    print("="*50)
    
    print(f"\n✅ Treinamento concluído!")
    print(f"🎯 Acurácia final: {test_accs[-1]*100:.2f}%")
    print(f"Tempo: {format_execution_time(time.time() - start_time)}")