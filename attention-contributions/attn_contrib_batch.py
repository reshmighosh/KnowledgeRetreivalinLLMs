import os
import argparse
import torch
import transformers 

from tqdm import tqdm
import numpy as np
import json
import random
import pickle

from model_lib.hf_tooling import HF_Llama2_Wrapper
from model_lib.attention_tools import run_attention_monitor
from factual_queries import load_constraint_dataset
from viz_tools import plot_attention_flow

import argparse
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/llama-2-7b-hf")
    parser.add_argument("--max_new_tokens", type=int, default=25, help="Number of tokens to generate for each prompt.")
    parser.add_argument("--dataset-name", type=str, default="known_1000_synthetic")
    parser.add_argument("--output-dir", type=str, default="./plots", help="Output directory to save the attention flow.")
    parser.add_argument("--json-output-file", type=str, help="Output directory to save generations")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("-f")

    return parser.parse_args()

args = config()

# Load the models
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, 
                                                          device_map="cuda", 
                                                          trust_remote_code=True, 
                                                          attn_implementation="eager",
                                                          cache_dir="/work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/.cache/")

model_wrapped = HF_Llama2_Wrapper(model, tokenizer, device="cuda")
items = load_constraint_dataset(args.dataset_name, filename=args.input_file)
items = sorted(items, key=lambda x: x["popularity"], reverse=True)

full_data = []
# stats = []

if not os.path.exists(f"{args.output_dir}"):
    os.makedirs(args.output_dir)

if not os.path.exists(f"{args.output_dir}/outputs/"):
    os.makedirs(f"{args.output_dir}/outputs/")

if not os.path.exists(f"{args.output_dir}/data"):
    os.makedirs(f"{args.output_dir}/data/")

if not os.path.exists(f"{args.output_dir}/plots/"):
    os.makedirs(f"{args.output_dir}/plots/")

print(f"A random prompt")
print(random.choice(items)['prompt'])


# Load the dataset to explore existing examples
for i,item in enumerate(tqdm(items)):

    constraints = []
    if 'counterfactual_object' in item.keys():
        # counterfactual RAG dataset
        constraints = [f" {item['counterfactual_object']}", f" {item['subject']}"]
    elif 'attribute' in item.keys():
        constraints = [f" {item['attribute']}", f" {item['subject']}"]
    else:
        # normal RAG dataset
        constraints = [f" {item['object']}", f" {item['subject']}"]

    prompt_info = {"prompt": item["prompt"], 
                    "constraints": constraints}

    data = run_attention_monitor(prompt_info,
                                model_wrapped)

    if data == dict():
        continue

    data['id'] = item['known_id']

    from viz_tools import plot_attention_flow

    start_idx = 1

    if 'phi' in args.model_name:
        start_idx = 0

    flow_matrix = data.all_token_contrib_norms[:, start_idx:data.num_prompt_tokens].T
    token_labels = data.token_labels[1:data.num_prompt_tokens]
    fig = plot_attention_flow(flow_matrix, token_labels, topk_prefix=512)
    fig.savefig(f"{args.output_dir}/plots/{data['id']}_{item['constraint']}.png", bbox_inches="tight")
    item['full_generation'] = data['full_prompt']
    item['completion'] = data['completion']
    pickle.dump(data, open(f"{args.output_dir}/data/stats_{data['id']}.pkl", "wb"))
    full_data.append(item)

json.dump(full_data, open(f"{args.output_dir}/outputs/{args.json_output_file}", 'w'))
