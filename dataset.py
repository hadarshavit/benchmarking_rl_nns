from torch.utils.data import Dataset
import torch
from torchvision.transforms import Resize, Grayscale
from tqdm import tqdm
import os
from collections import deque
import logging

class PolicyDataset(Dataset):
    def __init__(self, root_dir, frame_stack=4, resolution=84, grayscale=True) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_states = 0
        self.data = []
        
        self.frame_stack = frame_stack
        self.resolution = resolution
        
        self.resize = Resize(self.resolution)
        self.grayscale = Grayscale()

        for file in tqdm(os.listdir(self.root_dir)):
            full_path = os.path.join(self.root_dir, file)

            episode = torch.load(full_path)
            states = episode['states']
            print(states[0].shape)
            states = [self.grayscale(self.resize(state.permute(2, 0, 1))) for state in states]
            rewards = episode['rewards']
            actions = episode['actions']

            gt = 0

            for i in range(len(actions) - 1, -1, -1):
                gt = rewards[i] + 0.99 * gt

                state_buffer = deque([], maxlen=self.frame_stack)
                n_frames = min(i + 1, self.frame_stack)
                frames = states[i - n_frames + 1: i + 1]
                for _ in range(n_frames - self.frame_stack):
                    state_buffer.append(torch.zeros(84, 84, 1 if grayscale else 3, dtype=torch.uint8))

                for frame in frames:
                    state_buffer.append(frame)
                
                state_buffer = torch.stack(list(state_buffer), 0).unsqueeze(0).byte()

                self.data.append((state_buffer, gt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i][0], self.data[i][1]


