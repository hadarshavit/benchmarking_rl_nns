#!/bin/bash
#SBATCH --job-name=ablations3
#SBATCH --output=../out_files/%x_%j.out
#SBATCH --error=../out_files/%x_%j.err
#SBATCH --mail-user="h.shavit@umail.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=90G
#SBATCH --time=0-01:00:00
#SBATCH --partition=testing
#SBATCH --ntasks=8
#SBATCH --gpus=1
#SBATCH --nodes=1

echo "[$SHELL] #### Starting GPU TensorFlow test"
echo "[$SHELL] This is $SLURM_JOB_USER and my first job has the ID $SLURM_JOB_ID"
source /home/${USER}/.bashrc
conda activate /data1/s3092593/rlenv

echo $1