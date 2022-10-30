import argparse
from functools import partial
from dataset import PolicyDataset
import timm
from timm.utils import NativeScaler
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-path', type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda:0')
    amp_autocast = partial(torch.autocast, device_type=device.type, dtype=torch.float16)

    dataset = PolicyDataset(args.dataset_path)

    model = torch.nn.Embedding() # TODO

    criterion = torch.nn.SmoothL1Loss()
    mae = torch.nn.L1Loss()
    maes = torch.zeros(100)
    optimizer = torch.optim.Adam(model.parameters())
    train_loader = DataLoader(dataset, batch_size=256, num_workers=4)
    scaler = NativeScaler()

    for epoch in range(100):
        for data, target in train_loader:
            data.cuda()
            target.cuda()

            with amp_autocast():
                y = model(data)
                loss = criterion(y, target)
                maes[epoch] = mae(y, target)

            optimizer.zero_grad()
            scaler(loss, optimizer)

    torch.save(maes, 'maes.pt')


if __name__ == '__main__':
    main()