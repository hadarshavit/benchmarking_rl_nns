import os

if __name__ == '__main__':
    agents = ['DQN_modern']
    environments = ['Qbert']#, #'BattleZone', 'DoubleDunk', 'NameThisGame', 'Phoenix']
    checkpoints = ['model_15250000', 'model_37250000']#, 'model_50000000']

    for agent in agents:
        for environment in environments:
            for checkpoint in checkpoints:
                # if agent == 'DQN_modern' and environment == 'Qbert':
                    # continue
                os.system(f'sbatch prepare_dataset.slurm /home/s3092593/data1/atari_agents/ale_agents/{agent}/{environment}/0/{checkpoint}.gz {agent}_{environment}_{checkpoint}_0')