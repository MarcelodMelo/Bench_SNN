# datasets/base_loader.py
class BaseDataset:
    def __init__(self):
        self.metadata = {}  
        self.data = {}       
    
    def load_raw(self):
        # Implementado por cada dataset
        pass
    
    def get_metadata(self):
        return self.metadata.copy() 
    
    def get_data(self):
        return self.data  