#!/bin/bash
cd /HPS/NavidCAM/work/Creative_GAN/MONBO_Automized/
conda activate mina
sbatch Acquisition.sh
squeue -u nansari