#!/bin/bash
#SBATCH --job-name=hpo
#SBATCH --output=../out_files/%x_%j.out
#SBATCH --error=../out_files/%x_%j.err
#SBATCH --mem=90G
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu-long
#SBATCH --ntasks=6
#SBATCH --gpus=1
#SBATCH --nodes=1

source /home/${USER}/.bashrc
conda activate /data1/s3092593/rlenv

cp $1 /scratchdata/s3092593/${SLURM_JOB_ID}/data.zip
cd /scratchdata/s3092593/${SLURM_JOB_ID}
unzip -q data.zip
cd /home/s3092593/benchmarking_rl
# python train.py --dataset-path /scratchdata/s3092593/${SLURM_JOB_ID} --model $2 --reps 3 --lr 1e-3 --wandb-name hpo --use-validation True
# python train.py --dataset-path /scratchdata/s3092593/${SLURM_JOB_ID} --model $2 --reps 3 --lr 1e-4 --wandb-name hpo --use-validation True
# python train.py --dataset-path /scratchdata/s3092593/${SLURM_JOB_ID} --model $2 --reps 3 --lr 1e-2 --wandb-name hpo --use-validation True
# python train.py --dataset-path /scratchdata/s3092593/${SLURM_JOB_ID} --model $2 --reps 3 --lr 5e-3 --wandb-name hpo --use-validation True
# python train.py --dataset-path /scratchdata/s3092593/${SLURM_JOB_ID} --model $2 --reps 3 --lr 5e-4 --wandb-name hpo --use-validation True
# python train.py --dataset-path /scratchdata/s3092593/${SLURM_JOB_ID} --model $2 --reps 3 --lr 5e-2 --wandb-name hpo --use-validation True
python train.py --dataset-path /scratchdata/s3092593/${SLURM_JOB_ID} --model $2 --reps 3 --lr 1e-5 --wandb-name hpo --use-validation True
python train.py --dataset-path /scratchdata/s3092593/${SLURM_JOB_ID} --model $2 --reps 3 --lr 5e-5 --wandb-name hpo --use-validation True
python train.py --dataset-path /scratchdata/s3092593/${SLURM_JOB_ID} --model $2 --reps 3 --lr 1e-6 --wandb-name hpo --use-validation True
python train.py --dataset-path /scratchdata/s3092593/${SLURM_JOB_ID} --model $2 --reps 3 --lr 5e-6 --wandb-name hpo --use-validation True
# $1 data path $2 model name
