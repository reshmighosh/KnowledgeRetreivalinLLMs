#!/bin/bash
#SBATCH --job-name=0411_job_1
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=03:00:00

# Load the Conda module
module load miniconda/22.11.1-1 

# Activate the Conda environment
conda activate rome

# Change directory to your desired location
cd /work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/knowledge-perception/attn-flow

# Run the Python script
python norm_contribution.py