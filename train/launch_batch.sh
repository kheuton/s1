#!/bin/bash
#SBATCH --job-name=trains1
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=154G
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --output=log/sft_%j.txt  # %j is the job ID
uid="$(date +%Y%m%d_%H%M%S)"
./train/sft_slurm.sh --uid ${uid}

