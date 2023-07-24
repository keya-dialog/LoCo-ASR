import torch
import torch.nn as nn

class IntentClassifier(nn.Module):
    def __init__(self, method="average", embedding_size=768, num_heads=1):
        """
        Method can be:
        - average: mean of all values from frames / tokens
        - max: maximum of all values from  frames / tokens
        - self_attention: learn weights to select best values from frames / tokens
        """
        super(IntentClassifier, self).__init__()

        self.method = method
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        self.multihead_attention = nn.MultiheadAttention(self.embedding_size, self.num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(768,101),
        )

    def average(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        return avg
    
    def maximum(self, x):
        max = torch.max(x, dim=1, keepdim=True)
        return max 
    
    def self_attention(self, x):
        attention_outputs, attention_weights = self.multihead_attention(x,x,x)
        return attention_outputs #TO BE CHANGES

    def forward(self, x):
        #Downsample embeddings
        if self.method == "average":
            x = self.average(x)
        elif self.method == "max":
            x = self.maximum(x).values
        else:#self-attention
            x = self.self_attention(x)

        #classifier
        out = self.classifier(x)

        return out 