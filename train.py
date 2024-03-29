import argparse
from functools import partial
from dataset import PolicyDataset
import timm
from loader import PrefetchLoader, MultiEpochsDataLoader
import wandb
from timm.utils import NativeScaler
import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from architectures import get_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--prepare', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--wandb-name', type=str, default='')
    parser.add_argument('--use-validation', type=bool, default=False)
    parser.add_argument('--n_actions', type=int)

    return parser.parse_args()


def main(args, train_dataset, validation_dataset, test_dataset):
    n_epochs = 30
    timm.utils.random_seed(args.seed)
    device = torch.device('cuda:0')
    mname = args.model.replace(':', '_')

    if args.wandb:
        proj_name = args.wandb_name.replace(':', '_')
        wandb.init(project=f'benchmarkv2_{proj_name}_{mname}', config=args)

    amp_autocast = partial(torch.autocast, device_type=device.type, dtype=torch.float16)

    print('Creating Datasets')
    model = get_model(args.model, spectral_norm=False, resolution=(84, 84))(depth=4, actions=args.n_actions, linear_layer=nn.Linear).to(device)

    criterion = torch.nn.SmoothL1Loss()
    mae = torch.nn.L1Loss(reduction='sum')
    validation_maes = torch.zeros(n_epochs, device='cuda:0')
    test_maes = torch.zeros(n_epochs, device='cuda:0')
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    
    train_loader = PrefetchLoader(MultiEpochsDataLoader(train_dataset, batch_size=1024, num_workers=5, persistent_workers=True))
    validation_loader = PrefetchLoader(MultiEpochsDataLoader(validation_dataset, batch_size=1024, num_workers=5, persistent_workers=True))
    test_loader = PrefetchLoader(MultiEpochsDataLoader(test_dataset, batch_size=1024, num_workers=5, persistent_workers=True))
    scaler = NativeScaler()

    print('Training Starts!')
    for epoch in range(n_epochs):
        # total_loss = torch.zeros(1, device='cuda:0')
        for data, target, actions in train_loader:

            with amp_autocast():
                y = torch.gather(model(data), dim=1, index=actions)
                loss = criterion(y, target)
                # total_loss += loss
            optimizer.zero_grad()
            scaler(loss, optimizer)
        
        with torch.no_grad():
            for data, target, actions in validation_loader:
                with amp_autocast():
                    y = torch.gather(model(data), dim=1, index=actions)
                    validation_maes[epoch] += mae(y, target)

            for data, target, actions in test_loader:
                with amp_autocast():
                    y = torch.gather(model(data), dim=1, index=actions)
                    test_maes[epoch] += mae(y, target)

                    # if torch.isnan(maes[epoch]) or torch.isinf(maes[epoch]):
                    #     print('NAN or INF mae detectedm stopping training')
                    #     return 100

        validation_maes[epoch] /= len(validation_dataset)
        test_maes[epoch] /= len(test_dataset)
        if args.wandb:
            wandb.log({'loss': loss, 'validation_mae': validation_maes[epoch], 'test_mae': test_maes[epoch]})
        print(f'Epoch {epoch}, validation mae {validation_maes[epoch]}, test mae {test_maes[epoch]}')

    # torch.save(maes, 'maes.pt')
    
    # print('AUC:', torch.trapezoid(maes))
    if args.wandb:
        torch.save(model, f'/data1/s3092593/saved_benchmarks/benchmark_{args.wandb_name}_{mname}_{args.seed}.pt')
        artifact = wandb.Artifact('saved_model', type='model')
        artifact.add_file(f'/data1/s3092593/saved_benchmarks/benchmark_{args.wandb_name}_{mname}_{args.seed}.pt')
        wandb.log_artifact(artifact)
        wandb.finish()

    # return maes[-1].item()


if __name__ == '__main__':
    args = parse_args()

    train_dataset = PolicyDataset(os.path.join(args.dataset_path, 'train'), prepare=False)
    # if not args.use_validation:
    test_dataset = PolicyDataset(os.path.join(args.dataset_path, 'test'),  prepare=False)
    # else:
    validation_dataset = PolicyDataset(os.path.join(args.dataset_path, 'validation'),  prepare=False)

    for rep in range(args.reps):
        main(args, train_dataset, validation_dataset, test_dataset)
        args.seed += 1
