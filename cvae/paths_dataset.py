import random
from collections import deque, namedtuple

import numpy as np
import torch

from model_constants import CUDA_AVAILABLE
from torch.utils.data import Dataset
from model_constants import *

class PathsDataset(Dataset):
    """
    Example usage in a dataloader.
    this_dataset = ReplayBuffer(buffer_size)
    # Add some entries to this_dataset.
    dataloader = DataLoader(this_dataset, batch_size=128, shuffle=True, num_workers=4)
    # Now, simply iterate on this dataloader.
    for batch_idx, sampled_batch in enumerate(dataloader):
        # do stuff here using sampled_batch.
    """
    def __init__(self):
        self.num_paths = 0
        self.paths = []
        self.device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return {
            'env': self.paths[idx][0],
            'state': np.array(self.paths[idx][1:3]),
            'condition': np.array(self.paths[idx][3:]),
        }

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_paths < batch_size:
            paths = random.sample(self.paths, self.num_paths)
        else:
            paths = random.sample(self.paths, batch_size)

        env = torch.from_numpy(np.vstack([e[0] for e in paths if e is not None])).float().to(self.device)
        state = torch.from_numpy(np.vstack([e[1:] for e in paths if e is not None])).float().to(self.device)

        return (env, state)

    def add_path(self, state):
        self.paths.append(state)
    
    def add_env_paths(self, env_states):
        self.paths += env_states


