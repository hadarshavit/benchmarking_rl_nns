from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, ConfigurationSpace, Float, Categorical, EqualsCondition
from train import main
from dataset import PolicyDataset
import os
from argparse import ArgumentParser, Namespace

parser = ArgumentParser()
parser.add_argument('--dataset-path')
parser.add_argument('--model')

args = parser.parse_args()

train_dataset = PolicyDataset(os.path.join(args.dataset_path, 'train'))
train_dataset.preprocess()
test_dataset = PolicyDataset(os.path.join(args.dataset_path, 'test'))
test_dataset.preprocess()

def get_configuration_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(Float('lr', (1e-4, 1e-1), log=True))
    cs.add_hyperparameter(Float('weight_decay', (1e-12, 1e-1), log=True))
    cs.add_hyperparameter(Float('beta1', (0.9, 0.9999999), log=True))
    cs.add_hyperparameter(Float('beta2', (0.9, 0.9999999), log=True))
    
    return cs

def run_training(config, seed):
    train_args = Namespace(lr=config['lr'], beta1=config['beta1'], beta2=config['beta2'], 
                           weight_decay=config['weight_decay'], model=args.model)
    return main(train_args, train_dataset, test_dataset)

def run_smac():
    cs = get_configuration_space()
    scenario = Scenario(cs, n_trials=50, deterministic=True, output_directory=f'smac{args.model}')
    smac = HyperparameterOptimizationFacade(
        scenario,
        run_training,
        overwrite=True,  # If the run exists, we overwrite it; alternatively, we can continue from last state
    )
    incumbent = smac.optimize()

    print(incumbent)

if __name__ == '__main__':
    run_smac()