#!/bin/bash
#SBATCH --job-name=attn_contrib_job
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=12:00:00

# Load the Conda module
module load miniconda/22.11.1-1 

# Activate the Conda environment
conda activate rome

# Change directory to your desired location
cd /work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/repo-for-paper/attention-contributions/

# Run the Python script

python attn_contrib_batch_vanilla_only.py \
--dataset-name "known_1000_no_rag" \
--output-dir  /work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/repo-for-paper/attention-contributions/outputs/llama-2-base/run-vanilla \
--json-output-file full_data.json 