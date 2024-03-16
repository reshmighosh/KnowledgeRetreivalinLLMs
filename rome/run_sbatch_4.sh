#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram40
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --time=08:00:00

# Load the Conda module
module load miniconda/22.11.1-1 

# Activate the Conda environment
conda activate rome

# Change directory to your desired location
cd /work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/rome/

# Run the Python script
python -m experiments.causal_trace --fact_file /work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/rome/notebooks/incorrect_llama.jsonl --model_name llama --output_dir results/llama_incorrect/causal_trace --dont_expect