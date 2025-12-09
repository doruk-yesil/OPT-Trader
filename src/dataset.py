import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class BTCDataset(Dataset):
    def __init__(self, csv_path, seq_len=128):
        self.seq_len = seq_len
        
        df = pd.read_csv(csv_path)

        # Drop columns that should not be in training
        drop_cols = ["open_time", "close_time", "future_close", "future_return"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])

        # Replace NaN/inf with 0
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # Extract label
        self.y = df["label"].astype(np.int64).values
        
        # Remap labels from [-1,0,1] → [0,1,2]
        label_map = { -1: 0, 0: 1, 1: 2 }
        self.y = np.array([label_map[l] for l in self.y], dtype=np.int64)

        # Extract features
        self.X = df.drop(columns=["label"]).astype(np.float32).values

        self.len = len(df)

    def __len__(self):
        return self.len - self.seq_len

    def __getitem__(self, idx):
        seq = self.X[idx : idx + self.seq_len]
        label = self.y[idx + self.seq_len]

        # Return tensors directly → no dtype issues
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
