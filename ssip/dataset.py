import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import numpy.matlib as matlib
import torch, torchaudio
from torch.utils.data import Dataset
import os, random

class EEGLogMelDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir should contain:
        root_dir/eeg/*.npy
        root_dir/log_mel/*.npy
        """
        self.eeg_dir = os.path.join(root_dir, "eeg")
        self.log_mel_dir = os.path.join(root_dir, "log_mel")

        self.ids = sorted([f[:-4] for f in os.listdir(self.eeg_dir) if f.endswith(".npy")])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        eeg_path = os.path.join(self.eeg_dir, f"{self.ids[idx]}.npy")
        log_mel_path = os.path.join(self.log_mel_dir, f"{self.ids[idx]}.npy")

        eeg = np.load(eeg_path)  # (T_eeg, C_eeg)
        log_mel = np.load(log_mel_path)  # (n_mels, T_mel)

        # Convert to torch
        eeg_tensor = torch.from_numpy(eeg).float()
        log_mel_tensor = torch.from_numpy(log_mel).float()

        return eeg_tensor, log_mel_tensor
    
# Define the custom dataset class
class sEEG_Dataset(Dataset): 
    def __init__(self, X, y, va_labels, T, sr=100, n_segments_per_epoch=6400, augment = True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.va_labels = torch.tensor(va_labels, dtype=torch.float32) 
        self.sr = sr; self.T = T
        self.n_segments = n_segments_per_epoch
        self.augment = augment 
    def __len__(self):
        return self.n_segments # each epoch will draw n_segments_per_epoch random crops
    def __getitem__(self, idx):
        # Random start index for cropping
        start = random.randint(0, self.X.shape[0] - self.T*self.sr) ## self.X.shape[0] is the total length of the sample
        end = start + self.sr*self.T
        # Crop the same segment from both
        sEEG = self.X[start:end]
        mel = self.y[start:end] 
        va_label = self.va_labels[start:end]
        if self.augment:
            if random.random() < 0.5:
                noise = (0.316)*torch.randn(size = sEEG.shape, dtype=torch.float32) # variance of 0.1
                sEEG = sEEG + noise 
        return sEEG, mel, va_label 