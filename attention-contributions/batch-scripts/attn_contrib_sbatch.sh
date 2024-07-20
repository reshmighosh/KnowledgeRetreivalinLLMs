#!/bin/bash
#SBATCH --job-name=attn_contrib_job
#SBATCH --partition=gpu,gpu-preempt
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
cd /work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/repo-for-paper/attention-contributions-llama/

python attn_contrib_batch.py \
--dataset-name "known_1000_synthetic" \
--output-dir  /work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/repo-for-paper/attention-contributions-llama/outputs/llama-2-base/run-rag-synthetic-corrected-position-1 \
--json-output-file full_data.json \
--input-file /work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/repo-for-paper/data/Known_1209_RAG_data_with_object_at_1.json

python attn_contrib_batch.py \
--dataset-name "known_1000_synthetic" \
--output-dir  /work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/repo-for-paper/attention-contributions-llama/outputs/llama-2-base/run-rag-synthetic-corrected-position-2 \
--json-output-file full_data.json \
--input-file /work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/repo-for-paper/data/Known_1209_RAG_data_with_object_at_2.json
