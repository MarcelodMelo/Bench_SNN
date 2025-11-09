import torch
import snntorch.functional as SF
from .base_trainer import BaseTrainer

class BPTTTrainer(BaseTrainer):
    """Trainer para Backpropagation Through Time"""
    
    def __init__(self, model, optimizer, loss_fn=None, device='cpu'):
        # ✅ CORRIGIDO: Chama super() corretamente
        super().__init__(model, optimizer, loss_fn or SF.ce_rate_loss(), device)
        
    def train_epoch(self, dataloader):
        """Treina uma época completa"""
        self.model.train()
        total_loss, total_acc, num_batches = 0, 0, len(dataloader)
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            spk_rec, _ = self.model(data)
            
            # Loss e acurácia
            loss_val = self.loss_fn(spk_rec, targets)
            acc_val = SF.accuracy_rate(spk_rec, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
            
            total_loss += loss_val.item()
            total_acc += acc_val
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss_val.item():.4f}, Acc: {acc_val*100:.2f}%")
                
        return total_loss / num_batches, total_acc / num_batches
    
    def validate_epoch(self, dataloader):
        """Valida uma época"""
        self.model.eval()
        total_acc, num_batches = 0, len(dataloader)
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                spk_rec, _ = self.model(data)
                
                # Acurácia
                acc_val = SF.accuracy_rate(spk_rec, targets)
                total_acc += acc_val
                
        return total_acc / num_batches