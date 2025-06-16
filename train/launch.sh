#!/bin/bash
#SBATCH --job-name=trains1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=48
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=ccgpu
#SBATCH --time=1-00:00:00
#SBATCH --output=log/sft_%j.txt  # %j is the job ID

uid="$(date +%Y%m%d-%H%M%S)"
echo $HF_CACHE

srun --job-name=trains1 \
    --nodes=2 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --account=kheuto01 \
    --partition=ccgpu \
    --gres=gpu:4 \
    --mem=256G \
    --time=3-00:00:00 \
    ./train/sft_slurm.sh \
        --uid ${uid} \
        > log/sft_${uid}_interior.txt 2>&1 