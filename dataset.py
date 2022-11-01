from torch.utils.data import Dataset
import torch
from torchvision.transforms import Resize
from tqdm import tqdm
import os

class PolicyDataset(Dataset):
    def __init__(self, root_dir, frame_stack=4, resolution=84) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_states = 0
        self.data = []
        
        self.frame_stack = frame_stack
        self.resolution = resolution
        
        self.resize = Resize(self.resolution)

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

    def __getitem__(self, i):  # TODO: resize
        frames = [self.resize(d[0]) for d in self.data[i-self.frame_stack+1:i+1]] # TODO check order (compare with the atari_env.py's order)
        if len(frames) == 0 or self.frame_stack > len(self.data):
            logging.error("ERROR when stacking frames.") 
        return frames, self.data[i][1]


