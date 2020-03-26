#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=inception_v3-vae-pytorch
#SBATCH --output=slurm-logs/output_JOB=%j.txt

cd /scratch/$USER/inception-vae
source env/bin/activate
python3 train.py |& mail -s "[Prince] Job $SLURM_JOB_ID has finished running" $USER@nyu.edu
