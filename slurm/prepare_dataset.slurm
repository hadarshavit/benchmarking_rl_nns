#!/bin/bash
#SBATCH --job-name=perp-data
#SBATCH --output=../out_files/%x_%j.out
#SBATCH --error=../out_files/%x_%j.err
#SBATCH --mem=50G
#SBATCH --time=0-4:00:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=6
#SBATCH --gpus=1
#SBATCH --nodes=1

source /home/${USER}/.bashrc
conda activate /data1/s3092593/rlenv
mkdir /dev/shm/${SLURM_JOB_ID}
cd ..
python play.py $1 -f 700000 -d /dev/shm/${SLURM_JOB_ID}/train
python dataset.py /dev/shm/${SLURM_JOB_ID}/train
python play.py $1 -f 150000 -d /dev/shm/${SLURM_JOB_ID}/validation
python dataset.py /dev/shm/${SLURM_JOB_ID}/validation
python play.py $1 -f 150000 -d /dev/shm/${SLURM_JOB_ID}/test
python dataset.py /dev/shm/${SLURM_JOB_ID}/test

cd /dev/shm/${SLURM_JOB_ID}
zip -q -r data.zip train validation test
cp /dev/shm/${SLURM_JOB_ID}/data.zip /data1/s3092593/benchmark_dataset/$2.zip
rm -rf /dev/shm/${SLURM_JOB_ID}/data.zip
