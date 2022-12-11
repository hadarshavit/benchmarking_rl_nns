#!/bin/bash
#SBATCH --job-name=ablations3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="h.shavit@umail.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=90G
#SBATCH --time=0-02:00:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=6
#SBATCH --gpus=1
#SBATCH --nodes=1

echo "[$SHELL] #### Starting GPU TensorFlow test"
echo "[$SHELL] This is $SLURM_JOB_USER and my first job has the ID $SLURM_JOB_ID"
source /home/${USER}/.bashrc
conda activate /data1/s3092593/rlenv

python -m torch.utils.bottleneck play.py "/home/s3092593/data1/atari_agents/ALE agents/DQN_modern/Qbert/0/model_50000000.gz" -d /data1/s3092593/qbert_replays/DQN_modern/Qbert/0/model_50000000_new
# python 
