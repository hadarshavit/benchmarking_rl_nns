import argparse
from functools import partial
from dataset import PolicyDataset
import timm
from loader import PrefetchLoader, MultiEpochsDataLoader
import wandb
from timm.utils import NativeScaler
import torch
import os
from torch.utils.data import DataLoader
from architectures import get_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--beta1', type=float)
    parser.add_argument('--beta2', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--reps', type=int)
    parser.add_argument('--prepare', type=bool, default=True)

    return parser.parse_args()


def main(args, train_dataset, test_dataset):
    n_epochs = 300
    device = torch.device('cuda:0')
    mname = args.model.replace(':', '_')
    wandb.init(project=f'benchmark_qbert2_{mname}', config=args)
    amp_autocast = partial(torch.autocast, device_type=device.type, dtype=torch.float16)

    print('Creating Datasets')
    model = get_model(args.model, spectral_norm=False, resolution=(84, 84))
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
    maes = torch.zeros(n_epochs, device='cuda:0')
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    
    train_loader = PrefetchLoader(MultiEpochsDataLoader(train_dataset, batch_size=1024, num_workers=5, persistent_workers=True))
    test_loader = PrefetchLoader(MultiEpochsDataLoader(test_dataset, batch_size=1024, num_workers=5, persistent_workers=True))
    scaler = NativeScaler()

    print('Training Starts!')
    for epoch in range(n_epochs):
        for data, target in train_loader:
            data = data.cuda()
            target = target.cuda()

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
                    maes[epoch] += mae(y, target)

                    if torch.isnan(maes[epoch]) or torch.isinf(maes[epoch]):
                        return 100

        maes[epoch] /= len(test_dataset)
        wandb.log({'loss': loss, 'mae': maes[epoch]})
        print(f'Epoch {epoch}, mae {maes[epoch]}')

    torch.save(maes, 'maes.pt')
    
    print('AUC:', torch.trapezoid(maes))
    wandb.finish()

    return maes[-1].item()


if __name__ == '__main__':
    args = parse_args()

    train_dataset = PolicyDataset(os.path.join(args.dataset_path, 'train'))
    if args.prepare:
        train_dataset.preprocess()
    test_dataset = PolicyDataset(os.path.join(args.dataset_path, 'test'))
    if args.prepare:
        test_dataset.preprocess()

    for rep in range(args.reps):
        main(args, train_dataset, test_dataset)