import argparse
from functools import partial
from dataset import PolicyDataset
import timm
import wandb
from timm.utils import NativeScaler
import torch
import os
from torch.utils.data import DataLoader
from architectures import NatureCNN, ImpalaCNNSmall, ImpalaCNNLarge


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--model', type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda:0')
    wandb.init(project='benchmark', config=args)
    amp_autocast = partial(torch.autocast, device_type=device.type, dtype=torch.float16)

    print('Creating Datasets')
    train_dataset = PolicyDataset(os.path.join(args.dataset_path, 'train'))
    train_dataset.preprocess()
    test_dataset = PolicyDataset(os.path.join(args.dataset_path, 'test'))
    test_dataset.preprocess()

    if args.model == 'nature':
        model = NatureCNN(4, 1, torch.nn.Linear).cuda()
    elif args.model == 'imapala_small':
        model = ImpalaCNNSmall(4, torch.nn.Linear).cuda()
    elif args.model == 'impala_large:1':
        model = ImpalaCNNLarge(4, torch.nn.Linear, 1).cuda()
    elif args.model == 'impala_large:2':
        model = ImpalaCNNLarge(4, torch.nn.Linear, 2).cuda()

    criterion = torch.nn.SmoothL1Loss()
    mae = torch.nn.L1Loss(reduction='sum')
    maes = torch.zeros(100, device='cuda:0')
    optimizer = torch.optim.Adam(model.parameters())
    train_loader = DataLoader(train_dataset, batch_size=1024, num_workers=5, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, num_workers=5, persistent_workers=True)
    scaler = NativeScaler()

    print('Training Starts!')
    for epoch in range(100):
        for data, target in train_loader:
            # print(torch.min(data), torch.max(data), flush=True)
            data = data.cuda()
            target = target.cuda()
            # print(torch.min(target), torch.max(target), torch.mean(target))

            with amp_autocast():
                y = model(data)
                loss = criterion(y, target)

            optimizer.zero_grad()
            scaler(loss, optimizer)
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.cuda()
                target = target.cuda()
                with amp_autocast():
                    y = model(data)
                    # print(y, target)
                    # print(torch.min(target), torch.max(target), torch.mean(target))
                    # print(torch.min(y), torch.max(y), torch.mean(y))
                    maes[epoch] += mae(y, target)

        maes[epoch] /= len(test_dataset)
        wandb.log({'loss': loss, 'mae': maes[epoch]})
        print(f'Epoch {epoch}, mae {maes[epoch]}')

    torch.save(maes, 'maes.pt')
    
    print('AUC:', torch.trapezoid(maes))


if __name__ == '__main__':
    main()