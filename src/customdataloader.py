import config 
import pickle
import torch
from torch.utils.data import  Dataset

class dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        labels = self.data[1]
        Xdata  = self.data[0]
        self.Xdata = torch.tensor(Xdata).unsqueeze(1)
        self.labels = torch.tensor(labels)

    
    def __len__(self):
        return len(self.Xdata)

    def __getitem__(self, index):
        return self.Xdata[index], self.labels[index]



