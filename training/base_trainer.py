import torch
import time
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """Interface base para todos os trainers"""
    
    def __init__(self, model, optimizer, loss_fn, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
    @abstractmethod
    def train_epoch(self, dataloader):
        """Treina por uma época - deve ser implementado"""
        pass
        
    @abstractmethod
    def validate_epoch(self, dataloader):
        """Valida por uma época - deve ser implementado"""
        pass
        
    def train(self, train_loader, test_loader, num_epochs):
        """Loop principal de treinamento"""
        train_losses, train_accs, test_accs = [], [], []
        
        for epoch in range(num_epochs):
            print(f"--- Época {epoch+1}/{num_epochs} ---")
            
            # Treino
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validação
            test_acc = self.validate_epoch(test_loader)
            
            # Registrar métricas
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            print(f"Loss: {train_loss:.4f} | Acc Treino: {train_acc*100:.2f}% | Acc Teste: {test_acc*100:.2f}%")
            
        return train_losses, train_accs, test_accs