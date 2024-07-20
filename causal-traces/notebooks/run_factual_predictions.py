import torch
from tqdm import tqdm
import jsonlines
from util.globals import DATA_DIR
from math import ceil
from experiments.causal_trace import ModelAndTokenizer
from experiments.causal_trace import (
    predict_token, predict_topk
)
from dsets import KnownsDataset
import argparse

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Model checkpoint/HuggingFace model ID", default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--input-file", help="Input dataset path")
parser.add_argument("--output-file", help="Output file (JSONL) path")

args = parser.parse_args()

model_name = args.model
output_file = args.output_file
input_file = args.input_file

mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=False,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
)

knowns = KnownsDataset(known_loc=input_file)  # Dataset of known facts

def get_factual_predictions(prompts):
    tokens, probs = predict_token(mt, prompts, return_p=True)
    probs = probs.detach().cpu().numpy().tolist()
    return tokens, probs

def get_factual_predictions_top_k(prompts, k=5):
    tokens, probs = predict_topk(mt, prompts, k=k)
    return tokens, probs

tokens = []
probs = []
idxs =  []
batch_size = 8

for i in tqdm(range(0, int(ceil(len(knowns)/batch_size)))):
    batch = knowns[i*batch_size:min(len(knowns),(i+1)*batch_size)]
    prompts = [b['prompt'] for b in batch]
    record_ids = [b['known_id'] for b in batch]
    t,p = get_factual_predictions(prompts)
    tokens.extend(t)
    probs.extend(p)
    idxs.extend(record_ids)

prefix = ''

with jsonlines.open(output_file, "w") as writer:
    prompts = [prefix+k['prompt'] for k in knowns]
    attrs = [k['correct'] for k in knowns]
    for prompt, correct, pred, probability, kid in zip(prompts, attrs, tokens, probs, idxs):
        writer.write({"prompt": prompt, "correct": correct, "prediction": pred, "prob": probability, "known_id": kid})