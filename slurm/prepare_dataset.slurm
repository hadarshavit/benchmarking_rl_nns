#!/bin/bash
#SBATCH --job-name=perp-data
#SBATCH --output=../out_files/%x_%j.out
#SBATCH --error=../out_files/%x_%j.err
#SBATCH --mem=90G
#SBATCH --time=0-1:00:00
#SBATCH --partition=testing
#SBATCH --ntasks=12
#SBATCH --gpus=1
#SBATCH --nodes=1

source /home/${USER}/.bashrc
conda activate /data1/s3092593/rlenv
cd ..
python play.py $1 -f 7000 -d /scratchdata/s3092593/${SLURM_JOB_ID}/train
python dataset.py /scratchdata/s3092593/${SLURM_JOB_ID}/train
python play.py $1 -f 1500 -d /scratchdata/s3092593/${SLURM_JOB_ID}/validation
python dataset.py /scratchdata/s3092593/${SLURM_JOB_ID}/validation
python play.py $1 -f 1500 -d /scratchdata/s3092593/${SLURM_JOB_ID}/test
python dataset.py /scratchdata/s3092593/${SLURM_JOB_ID}/test

cd /scratchdata/s3092593/${SLURM_JOB_ID}
zip -q -r data.zip train validation test
cp /scratchdata/s3092593/${SLURM_JOB_ID}/data.zip /data1/s3092593/benchmark_dataset/$2.zip

