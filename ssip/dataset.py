import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import numpy.matlib as matlib
import torch, torchaudio
from torch.utils.data import Dataset
import os 

class EEGMelDataset(Dataset):
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