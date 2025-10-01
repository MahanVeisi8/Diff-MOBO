#!/bin/bash
#SBATCH -p gpu20 
#SBATCH -t 1:20:00
#SBATCH -o slurm-%j.out
#SBATCH --gres gpu:1
#SBATCH --array=1
#SBATCH -o      Models/slurm_log/slurm-%x-%j-%a.log
#SBATCH --error Models/slurm_log/slurm-%x-%j-%a.err
n_iter=0
cd /HPS/NavidCAM/work/Creative_GAN/MONBO_Automized
ls
python Constraint_handler.py -iter_num "$n_iter"

