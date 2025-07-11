#!/bin/bash
cd /HPS/NavidCAM/work/Creative_GAN/MONBO_Automized
conda activate mina
sbatch train_constraint_handler.sh
squeue -u nansari