#!/bin/bash
#SBATCH --job-name=ablations3
#SBATCH --output=../out_files/%x_%j.out
#SBATCH --error=../out_files/%x_%j.err
#SBATCH --mail-user="h.shavit@umail.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem=90G
#SBATCH --time=7-0:00:00
#SBATCH --partition=gpu-long
#SBATCH --ntasks=6
#SBATCH --gpus=1
#SBATCH --nodes=1

echo "[$SHELL] #### Starting GPU TensorFlow test"
echo "[$SHELL] This is $SLURM_JOB_USER and my first job has the ID $SLURM_JOB_ID"
source /home/${USER}/.bashrc
conda activate /data1/s3092593/rlenv

# cp -r /data1/s3092593/qbert_replays/DQN_modern/Qbert/0/model_50000000_new /dev/shm/dataset_{$SLURM_JOB_USER}
cp /data1/s3092593/qbert_replays/DQN_modern/Qbert/0/model_50000000_new/data.zip /dev/shm/dataset_$SLURM_JOB_ID.zip
mkdir /dev/shm/dataset_$SLURM_JOB_ID
mv /dev/shm/dataset_$SLURM_JOB_ID.zip /dev/shm/dataset_$SLURM_JOB_ID/data.zip
cd /dev/shm/dataset_$SLURM_JOB_ID
unzip -q data.zip
cd /home/s3092593/benchmarking_rl
python train.py --dataset-path /dev/shm/dataset_$SLURM_JOB_ID --model impala_large:2 --beta1 0.9 --beta2 0.999 --lr 1e-3 --weight_decay 0 --reps 3 --prepare True
python train.py --dataset-path /dev/shm/dataset_$SLURM_JOB_ID --model impala_large:2 --beta1 0.9 --beta2 0.999 --lr 1e-4 --weight_decay 0 --reps 3 --prepare False
python train.py --dataset-path /dev/shm/dataset_$SLURM_JOB_ID --model impala_large:2 --beta1 0.9 --beta2 0.999 --lr 1e-2 --weight_decay 0 --reps 3 --prepare False
python train.py --dataset-path /dev/shm/dataset_$SLURM_JOB_ID --model impala_large:2 --beta1 0.9 --beta2 0.999 --lr 5e-3 --weight_decay 0 --reps 3 --prepare False
python train.py --dataset-path /dev/shm/dataset_$SLURM_JOB_ID --model impala_large:2 --beta1 0.9 --beta2 0.999 --lr 5e-2 --weight_decay 0 --reps 3 --prepare False
python train.py --dataset-path /dev/shm/dataset_$SLURM_JOB_ID --model impala_large:2 --beta1 0.9 --beta2 0.999 --lr 5e-4 --weight_decay 0 --reps 3 --prepare False




rm -rf /dev/shm/dataset_$SLURM_JOB_ID