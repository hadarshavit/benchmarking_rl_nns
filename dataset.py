from torch.utils.data import Dataset
import torch
from torchvision.transforms import Resize, Grayscale
from tqdm import tqdm
import os
from torch import nn
from collections import deque
import logging
import torch

class PolicyDataset(Dataset):
    def __init__(self, root_dir, frame_stack=4, resolution=84, grayscale=True, prepare=False) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_states = 0
        self.data = []
        
        self.frame_stack = frame_stack
        self.resolution = resolution
        
        self.resize = Resize((self.resolution, self.resolution))
        self.grayscale = Grayscale() if grayscale else nn.Identity()
        self.n_channels = 3 if not grayscale else 1

        if prepare:
            self.prepare()
        else:
            self.load()


    def prepare(self):
        n = 0
        for file in tqdm(os.listdir(self.root_dir)):
            n+= 1
            # if n == 5:
                # break
            print(file)
            full_path = os.path.join(self.root_dir, file)

            episode = torch.load(full_path)
            states = episode['states']
            # print(states[0].shape)
            states = [self.grayscale(self.resize(state.permute(2, 0, 1))) for state in states]
            rewards = episode['rewards']
            actions = episode['actions']

            gt = 0

            episode_data = []
            for i in range(len(actions) - 1, -1, -1):
                # import pdb; pdb.set_trace()
                reward = rewards[i]
                clipped = max(min(reward, 1), -1)
                gt = clipped + 0.99 * gt

                state_buffer = torch.zeros(self.frame_stack, self.n_channels, 84, 84, dtype=torch.int8)
                
                n_frames = min(i + 1, self.frame_stack)

                frames = torch.stack(states[i - n_frames + 1: i + 1])
                state_buffer[self.frame_stack - frames.shape[0]: ] = frames
                
                state_buffer = state_buffer.byte().reshape((self.n_channels * self.frame_stack, 84, 84))

                episode_data.append((state_buffer, gt))
            
            self.data += episode_data

        torch.save(self.data, os.path.join(self.root_dir, 'processed.pt'))

    def load(self):
        self.data = torch.load(os.path.join(self.root_dir,'processed.pt'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i][0].half().div(255), self.data[i][1].unsqueeze(0)


if __name__ == '__main__':
    PolicyDataset('/data1/s3092593/qbert_replays/DQN_modern/Qbert/0/model_50000000/train', prepare=True)
    PolicyDataset('/data1/s3092593/qbert_replays/DQN_modern/Qbert/0/model_50000000/test', prepare=True)