import random
from collections import deque, namedtuple

import numpy as np
import torch

from model_constants import CUDA_AVAILABLE
from torch.utils.data import Dataset


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
    def __init__(self, buffer_size, with_replacement=False):
        self.num_paths = 0
        self.buffer = []
        self.paths = namedtuple('Paths', field_names=['env', 'state'])
        self.device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return {
            'env': self.buffer[idx].env,
            'state': self.buffer[idx].state,
        }

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_paths < batch_size:
            paths = random.sample(self.buffer, self.num_paths)
        else:
            paths = random.sample(self.buffer, batch_size)

        env = torch.from_numpy(np.vstack([e.env for e in paths if e is not None])).float().to(self.device)
        state = torch.from_numpy(np.vstack([e.state for e in paths if e is not None])).float().to(self.device)

        return (env, state)

    def add(self, x, condition):
        path = self.paths(x, condition)
        self.buffer.append(path)


