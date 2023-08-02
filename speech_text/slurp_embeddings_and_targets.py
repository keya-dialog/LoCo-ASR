import os
import pickle
import torch
from torch.utils.data import Dataset

class SLURPEmbeddingsTargets(Dataset):
    def __init__(self, data_path, modality="text", split="train"):
        self.data_path = data_path
        self.full_path = os.path.join(os.path.join(data_path, split), modality)
        self.dataset = self.load_data()

    def load_data(self):
        files = os.listdir(self.full_path)
        return files
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        with open(os.path.join(self.full_path, self.dataset[idx]), "rb") as file:
            embeddings_and_targets = pickle.load(file)

        slurp_id = embeddings_and_targets["id"]
        embedding = torch.from_numpy(embeddings_and_targets["embedding"])
        target = torch.from_numpy(embeddings_and_targets["target"])
        
        return slurp_id, embedding, target
