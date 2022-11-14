from torch.utils.data import Dataset
import torch
from torchvision.transforms import Resize, Grayscale, PILToTensor, Compose, ConvertImageDtype
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
import os
from torch import nn
from collections import deque
import logging
from PIL import Image
import torch

class PolicyDataset(Dataset):
    def __init__(self, root_dir, frame_stack=4, resolution=84, transforms=None, prepare=False, clip_rewards=1) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.num_states = 0
        self.clip_rewards = clip_rewards
        self.data = []
        
        self.frame_stack = frame_stack
        self.resolution = resolution
        self.n_channels = 1
        
        if transforms == None:
            self.transforms = Compose([
                Resize((self.resolution, self.resolution)),
                Grayscale(),
                PILToTensor(),
                ConvertImageDtype(torch.float16)
            ])
        else:
            self.transforms = transforms
        
        if prepare:
            self.prepare()
        else:
            self.load()


    def prepare(self):
        data = torch.load(f'{self.root_dir}/data.pt')
        processed_data = []
        for e_id, rewards, actions in tqdm(data):
            gt = 0

            for i in range(len(rewards) - 1, -1, -1):
                # import pdb; pdb.set_trace()
                reward = rewards[i]
                action = actions[i]
                clipped = max(min(reward, self.clip_rewards), -self.clip_rewards)
                gt = clipped + 0.99 * gt

            
                processed_data.append((e_id, i, gt, action))

        torch.save(processed_data, os.path.join(self.root_dir, 'processed.pt'))

    def preprocess(self):
        for i in tqdm(range(len(self.data))):
            eid, fid, gt, action = self.data[i]
            img = Image.open(f'{self.root_dir}/e{eid}_f{fid}.png')
            img = self.transforms(img)
            torch.save(img, f'{self.root_dir}/e{eid}_f{fid}.pt')

    def load(self):
        self.data = torch.load(os.path.join(self.root_dir, 'processed.pt'))

    def __len__(self):
        return len(self.data)

    def get_image(self, i):
        eid, fid, gt, action = self.data[i]
        # img = Image.open(f'{self.root_dir}/e{eid}_f{fid}.png')
        # img = self.transforms(img)
        return torch.load(f'{self.root_dir}/e{eid}_f{fid}.pt')
        # return pil_to_tensor(img)

    def __getitem__(self, i):
        eid, fid, gt, action = self.data[i]
        state_buffer = torch.zeros(self.frame_stack, self.n_channels, 84, 84, dtype=torch.half)
        c = 3
        for j in range(fid, max(-1, fid - self.frame_stack), -1):
            state_buffer[c] = self.get_image(i - fid + j)
            c -= 1
        
        return state_buffer.reshape((self.n_channels * self.frame_stack, 84, 84)), torch.FloatTensor([gt]), torch.LongTensor([action])


if __name__ == '__main__':
    
    # PolicyDataset('/data1/s3092593/qbert_replays/DQN_modern/Qbert/0/model_50000000_new/train', prepare=True)
    PolicyDataset('/data1/s3092593/qbert_replays/DQN_modern/Qbert/0/model_50000000_new/test', prepare=True)