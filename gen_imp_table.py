import wandb
import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt, operator as op
import numpy as np
def print_maes():
    data = np.load('all_runs_data.npy', allow_pickle=True).item()

    agents = ['MDQN_modern', 'DQN_modern'] # DQN_modern
    environments = ['Qbert', 'BattleZone', 'DoubleDunk', 'NameThisGame', 'Phoenix']
    checkpoints = ['model_15250000', 'model_37250000', 'model_50000000']
    networks = [('nature', 0.001), 
                ('dueling', 0.001),
                ('impala_large:1', 0.0005),
                ('impala_large:2', 0.0005),
                ('impala_large:4', 0.0001),
                # ('impalanextv2_large:1', 0.0001),
                ('impalanextv2_large:2', 0.00001)]
    for environment in environments:
        i = 0
        for checkpoint in checkpoints:
                vals = []
                for model, lr in networks:
                    maes = []
                    for agent in agents:
                        
                        succ = 0
                        project_name = f'benchmark_final_{model}_{agent}_{environment}_{checkpoint}_0_{model}'.replace(':', '_')
                        
                        for run in data[project_name]:
                            
                            try:
                                # if model == 'impalanextv2_large:2':
                                #     # print(run.config['lr'])
                                #     if run.config['lr'] == 0.0001:
                                #         continue
                                maes.append(run.iloc[29]['mae'])
                                succ += 1
                                if succ == 10:
                                    break
                            except Exception as e:
                                print(project_name, e)
                        if succ < 10:
                            print(project_name, succ)
                    vals.append(f'${np.mean(maes):.2f} \pm {np.std(maes):.2f}$')
                if checkpoint == 'model_15250000':
                    check_name = '61M'
                elif checkpoint == 'model_37250000':
                    check_name = '149M'
                else:
                    check_name = '200M'
                if i == 0:
                    print('\multirow{3}{*}{' + environment + '} & ',check_name, '&',' & '.join(vals), '\\\\')
                else: 
                    print('{} & ',check_name, '&',' & '.join(vals), '\\\\')
                if i == 2:
                    print('\hline')
                i += 1

def print_imps():
    data = np.load('all_runs_data.npy', allow_pickle=True).item()

    agents = ['MDQN_modern', 'DQN_modern'] # DQN_modern
    environments = ['Qbert', 'BattleZone', 'DoubleDunk', 'NameThisGame', 'Phoenix']
    # environments = ['Phoenix']
    checkpoints = ['model_15250000', 'model_37250000', 'model_50000000']
    networks = [('nature', 0.001), 
                ('dueling', 0.001),
                ('impala_large:1', 0.0005),
                ('impala_large:2', 0.0005),
                ('impala_large:4', 0.0001),
                # ('impalanextv2_large:1', 0.0001),
                ('impalanextv2_large:2', 0.00001)]
    per_model = {'nature': [], 'dueling': [], 'impala_large:1': [], 'impala_large:2': [], 'impala_large:4': [], 'impalanextv2_large:2': []}
    for environment in environments:
        i = 0
        for checkpoint in checkpoints:
                vals = []
                base = {'MDQN_modern': 0, 'DQN_modern': 0}
                for model, lr in networks:
                    maes = []
                    imps = []
                    
                    for agent in agents:
                        
                        succ = 0
                        project_name = f'benchmark_final_{model}_{agent}_{environment}_{checkpoint}_0_{model}'.replace(':', '_')
                        
                        for run in data[project_name]:
                            
                            try:
                                # if model == 'impalanextv2_large:2':
                                #     # print(run.config['lr'])
                                #     if run.config['lr'] == 0.0001:
                                #         continue
                                maes.append(run.iloc[29]['mae'])
                                succ += 1
                                if succ == 10:
                                    break
            
                            except Exception as e:
                                print(project_name, e)
                        if succ < 10:
                            print(project_name, succ)
                        if model == 'nature':
                            base[agent] = np.mean(maes)
                        imps.append((base[agent] - pd.Series(maes).mean()) / base[agent] * 100)
                        per_model[model].append((base[agent] - pd.Series(maes).mean()) / base[agent] * 100)
                        # print(model, (base[agent] - pd.Series(maes).mean()) / base[agent] * 100, np.mean(maes), base[agent])
                    if model != 'nature':
                        vals.append(f'${np.mean(imps):.2f}$')
                if checkpoint == 'model_15250000':
                    check_name = '61M'
                elif checkpoint == 'model_37250000':
                    check_name = '149M'
                else:
                    check_name = '200M'

                if i == 0:
                    print('\multirow{3}{*}{' + environment + '} & ',check_name, '&',' & '.join(vals), '\\\\')
                else: 
                    print('{} & ',check_name, '&',' & '.join(vals), '\\\\')
                if i == 2:
                    print('\hline')
                i += 1
    vals = []
    for model in per_model:
        if model != 'nature':
            vals.append(f'${np.mean(per_model[model]):.2f}$')
    print('Average & ', '&',' & '.join(vals), '\\\\')
print_imps()