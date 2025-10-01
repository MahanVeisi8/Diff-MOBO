#!/bin/bash
cd /HPS/NavidCAM/work/Creative_GAN/MONBO_Automized
conda activate mina
sbatch Forward_UANA_diverse_activation.sh
squeue -u nansari