import os

if __name__ == '__main__':
    agents = ['MDQN_modern']
    environments = ['Qbert', 'BattleZone', 'DoubleDunk', 'NameThisGame', 'Phoenix']
    checkpoints = ['model_15250000', 'model_37250000', 'model_50000000']
    networks = [('dueling', 0.001),
                ('nature', 0.001),
                ('impala_large:1', 0.0005),
                ('impala_large:2', 0.0005),
                ('impala_large:4', 0.0001),
                ('impalanextv2_large:2', 0.0001),
                ('impalanextv2_large:2', 0.00001)]

    for agent in agents:
        for environment in environments:
            for checkpoint in checkpoints:
                for network, lr in networks:
                    # $1 data path $2 model name $3 game&checkpoint $4 lr
                    os.system(f'sbatch final_experiments.slurm /home/s3092593/data1/benchmark_dataset/{agent}_{environment}_{checkpoint}_0.zip {network} {agent}_{environment}_{checkpoint}_0 {lr}')
