#!/bin/bash
#SBATCH --job-name=hpo
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mem=90G
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu-medium
#SBATCH --ntasks=6
#SBATCH --gpus=1
#SBATCH --nodes=1

source /home/${USER}/.bashrc
conda activate /data1/s3092593/rlenv

cp $1 /scratchdata/s3092593/${SLURM_JOB_ID}/data.zip
unzip -q data.zip
cd /home/s3092593/benchmarking_rl
python train.py --dataset-path /scratchdata/s3092593/${SLURM_JOB_ID} --model $2 --reps 3 --lr $4 --wandb-name final_$2_$3 --use-validation False

# $1 data path $2 model name $3 game $4 lr
