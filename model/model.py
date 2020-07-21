import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch

class TabularModel(BaseModel):
    def __init__(self, embedding_sizes, n_continuous):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_continuous = n_emb, n_continuous
        self.linear1 = nn.Linear(self.n_emb + self.n_continuous, 200)
        self.linear2 = nn.Linear(200, 70)
        self.linear3 = nn.Linear(70, 5)
        self.bn1 = nn.BatchNorm1d(self.n_continuous)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(70)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)
        
    def forward(self, x_cat, x_cont):
        
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.linear1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.linear2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.linear3(x)
        return x
