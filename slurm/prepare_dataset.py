import os

if __name__ == '__main__':
    agents_paths = []
    save_paths = []

    for agent, save_path in zip(agents_paths, save_paths):
        os.system(f'python play.py {agent} -f 700000 -d {save_path}/train && python dataset.py --root_dir ')
        os.system(f'python play.py {agent} -f 700000 -d {save_path}/validation')
        os.system(f'python play.py {agent} -f 700000 -d {save_path}/test')