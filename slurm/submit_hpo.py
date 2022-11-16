import os

if __name__ == '__main__':
    lrs = [1e-3, 1e-4, 1e-2, 5e-3, 5e-4, 5e-2]
    reps = [1, 2, 3]
    nn_names = ['impala_large:1', 'impala_large:2', 'convnext_atto', 'nature', 'impalanext']
    dataset_path = '' # TODO

    for lr in lrs:
        for rep in reps:
            for nn_name in nn_names:
                os.system(f'sbatch -p gpu-medium --cpus-per-task 6 --gpus 1 --mem 90G --wrap \"python train.py --seed {rep} --model {nn_name} --dataset-path {dataset_path} --lr {lr}\"')