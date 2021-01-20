from torchvision import datasets, transforms
import torch
import pandas as pd
from base import BaseDataLoader, TabularDatasets
    
class TabularDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.data = pd.read_csv(data_dir, encoding="utf-8")
        self.dataset = TabularDatasets(self.data)
        self.dataset.missing_data()
        self.dataset.label_encoding()
        self.dataset.make_alldata_category()
        self.embedding_sizes, self.n_numeric = self.dataset.choosing_embedded_columns()
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
