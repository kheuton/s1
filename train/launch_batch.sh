#!/bin/bash
#SBATCH --job-name=trains1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=48
#SBATCH --nodes=2
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --partition=ccgpu
#SBATCH --time=3-00:00:00
#SBATCH --output=log/sft_%j.txt  # %j is the job ID
uid="$(date +%Y%m%d_%H%M%S)"
./train/sft_slurm.sh --uid ${uid}

