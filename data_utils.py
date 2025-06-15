"""
This script contains data preparation and data loader for ASVspoof2019 LA database.
With enhancements for AASIST2: Dynamic Chunk Size (DCS) and duration based margin.
"""

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import os

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


def dynamic_chunk_size(x: np.ndarray, min_samples: int = 16000, max_samples: int = 96000):
    """
    Dynamically truncates or pads an audio utterance to a random length between min_samples and max_samples.
    
    Args:
        x (np.ndarray): Input audio samples
        min_samples (int): Minimum number of samples (e.g., 16000 for 1 second at 16kHz)
        max_samples (int): Maximum number of samples (e.g., 96000 for 6 seconds at 16kHz)
        
    Returns:
        np.ndarray: Dynamically sized audio samples
        float: Duration in seconds (samples / 16000)
    """
    x_len = x.shape[0]
    
    # Randomly choose a target length between min and max samples
    target_len = np.random.randint(min_samples, max_samples + 1)
    
    # Calculate duration in seconds (assuming 16kHz sampling rate)
    duration = target_len / 16000
    
    if x_len >= target_len:
        # If the audio is longer than target, randomly crop it
        stt = np.random.randint(0, x_len - target_len + 1)
        return x[stt:stt + target_len], duration
    else:
        # If the audio is shorter than target, repeat and pad
        num_repeats = int(target_len / x_len) + 1
        padded_x = np.tile(x, num_repeats)[:target_len]
        return padded_x, duration


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self,
                 list_IDs,
                 base_dir,
                 dcs=False,  # Enable dynamic chunk size
                 min_samples=16000,  # Minimum samples (1s at 16kHz)
                 max_samples=96000,  # Maximum samples (6s at 16kHz)
                 ):
        """
        Generate a dataset for training
        
        Args:
            list_IDs (list): List of file names to use
            base_dir (str): Path to dataset directory
            dcs (bool): Whether to use dynamic chunk size
            min_samples (int): Minimum samples for DCS (default: 16000 = 1 second)
            max_samples (int): Maximum samples for DCS (default: 96000 = 6 seconds)
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.dcs = dcs
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.label_dict = {
            'spoof': 1,
            'bonafide': 0
        }

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = sf.read(os.path.join(self.base_dir, utt_id))
        
        # Apply dynamic chunk size if enabled
        if self.dcs:
            X, duration = dynamic_chunk_size(X, self.min_samples, self.max_samples)
        else:
            X = pad_random(X)
            duration = len(X) / 16000  # Calculate duration in seconds
            
        # Get label
        y = self.label_dict[utt_id.split('/')[1].split('-')[4].split('.')[0]]
        
        # Convert data to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor([y])
        duration_tensor = torch.FloatTensor([duration])
        
        return X_tensor, y_tensor, duration_tensor


class Dataset_ASVspoof2019_deveval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.label_dict = {
            'spoof': 1,
            'bonafide': 0
        }

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = sf.read(os.path.join(self.base_dir, utt_id))
        X = pad(X)
        y = self.label_dict[utt_id.split('/')[1].split('-')[4].split('.')[0]]
        return torch.FloatTensor(X), torch.LongTensor([y]), utt_id
