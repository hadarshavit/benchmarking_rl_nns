import argparse
from functools import partial
from dataset import PolicyDataset
import timm
import wandb
from timm.utils import NativeScaler
import torch
import os
from torch.utils.data import DataLoader
from architectures import NatureCNN


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-path', type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda:0')
    wandb.init(project='benchmark', config=args)
    amp_autocast = partial(torch.autocast, device_type=device.type, dtype=torch.float16)

    train_dataset = PolicyDataset(os.path.join(args.dataset_path, 'train'))
    test_dataset = PolicyDataset(os.path.join(args.dataset_path, 'test'))

    model = NatureCNN(4, 1, torch.nn.Linear)

    criterion = torch.nn.SmoothL1Loss()
    mae = torch.nn.L1Loss()
    maes = torch.zeros(100)
    optimizer = torch.optim.Adam(model.parameters())
    train_loader = DataLoader(train_dataset, batch_size=256, num_workers=4)
    test_loader = DataLoader(test_loader, batch_size=256, num_workers=4)
    scaler = NativeScaler()

    for epoch in range(100):
        for data, target in train_loader:
            data.cuda()
            target.cuda()

            with amp_autocast():
                y = model(data)
                loss = criterion(y, target)

            optimizer.zero_grad()
            scaler(loss, optimizer)
        
        with torch.no_grad():
            for data, target in test_loader:
                data.cuda()
                target.cuda()
                with amp_autocast():
                    y = model(data)
                    maes[epoch] += mae(y, target)

        wandb.log({'loss': loss, 'mae': mae[epoch]})

    torch.save(maes, 'maes.pt')
    
    print('AUC:', torch.trapezoid(maes))


if __name__ == '__main__':
    main()