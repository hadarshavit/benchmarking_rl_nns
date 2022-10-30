from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import os

class PolicyDataset(Dataset):
    def __init__(self, root_dir) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_states = 0
        self.data = []

        for file in tqdm(os.listdir(self.root_dir)):
            full_path = os.path.join(self.root_dir, file)

            episode = torch.load(full_path)
            states = episode['states']
            rewards = episode['rewards']
            actions = episode['actions']

            gt = 0

            for i in range(len(actions) - 1, -1, -1):
                gt = rewards[i] + 0.99 * gt

                self.data.append((states[i], gt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):  # TODO: frame stacking, resize
        return self.data[i]

