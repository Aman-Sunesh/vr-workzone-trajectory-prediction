import numpy as np
import torch
from torch.utils.data import Dataset


class WorkZoneTrajectoryDataset(Dataset):
    def __init__(self, npz_path: str, split: str = None):
        data = np.load(npz_path, allow_pickle=True)

        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

        if split is not None:
            splits = data["split"].astype(str)
            mask = splits == split

            X = X[mask]
            y = y[mask]

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        past = torch.from_numpy(self.X[idx])
        future = torch.from_numpy(self.y[idx])
        return past, future