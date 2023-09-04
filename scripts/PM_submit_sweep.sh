#!/bin/bash

#SBATCH -A m3443_g -q regular
#SBATCH -C gpu 
#SBATCH -t 0:30:00
#SBATCH -n 2
#SBATCH --ntasks-per-node=2
#SBATCH -c 64
#SBATCH --gpus-per-task=1
#SBATCH -o logs/%x-%j.out
#SBATCH -J JetGravNetSweep

eval "$(conda shell.bash hook)"

conda activate exatrkx-cori
export SLURM_CPU_BIND="cores"
echo -e "\nStarting sweeps\n"

for i in {0..1}; do
    echo "Launching task $i"
    srun --exact -u -n 1 --gpus-per-task 1 -c 64 --mem-per-gpu=110G wandb agent murnanedaniel/LRP_JetTagging_Study_B_sweep/rqpvnqft &
done
wait