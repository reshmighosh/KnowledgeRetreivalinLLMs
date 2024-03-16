python run_factual_predictions.py --model google/gemma-7b  --output-file predictions_gemma_top5.jsonl --input-file data_gemma.jsonl
python run_factual_predictions.py --model microsoft/phi-2  --output-file predictions_phi_top5.jsonl --input-file data_phi.jsonl
python run_factual_predictions.py --model tiiuae/falcon-7b  --output-file predictions_falcon_top5.jsonl --input-file data_falcon.jsonl
python run_factual_predictions.py --model mistralai/Mistral-7B-v0.1  --output-file predictions_mistral_top5.jsonl --input-file data_mistral.jsonl
python run_factual_predictions.py --model /work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/llama-2-7b-hf  --output-file predictions_llama_top5.jsonl --input-file data_llama.jsonl
