import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class TabularDatasets(Dataset):
    
    def __init__(self, data):
        
        self.cat_columns = [s for s in list(data.select_dtypes(exclude=["number"]).columns)]
        self.num_columns = [s for s in list(data.select_dtypes(include=["number"]).columns) if (s != 'Target') & (s != 'ID')]
        self.id_target_columns = ["ID", "Target"]
        
        self.X = data.drop(["ID", "Target"], axis=1)
        self.X1 = 0#data.loc[:, cat_columns].copy().values # categorical
        self.X2 = 0#data.loc[:, num_columns].copy().values # numerical
        self.y = data["Target"].values

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]
    
    def missing_data(self):
        for col in self.X.columns:
            if self.X[col].isnull().sum() > 10000:
                print("dropping", col, X[col].isnull().sum())
                self.X = self.X.drop(col, axis=1)
    
    def label_encoding(self):
        for col in self.X.columns:
            if self.X.dtypes[col] == "object":
                self.X[col] = self.X[col].fillna("NA")
            else:
                self.X[col] = self.X[col].fillna(self.X[col].mean())
            self.X[col] = LabelEncoder().fit_transform(self.X[col])
            self.y = LabelEncoder().fit_transform(self.y)
    
    def make_alldata_category(self):        
        for col in self.X.columns:
            self.X[col] =self.X[col].astype('category')
            
    def separate_data(self):        
        self.X1 = self.X.loc[:, self.cat_columns].copy().values.astype(np.int64) # categorical
        self.X2 = self.X.loc[:, self.num_columns].copy().values.astype(np.float32) # numerical
        
    def choosing_embedded_columns(self):
        embedded_cols = {n: len(col.cat.categories) for n,col in self.X[self.cat_columns].items() if len(col.cat.categories) > 2}
        embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]
        num_size = len(self.X.columns)-len(embedded_cols)
        
        return embedding_sizes, num_size
            
    