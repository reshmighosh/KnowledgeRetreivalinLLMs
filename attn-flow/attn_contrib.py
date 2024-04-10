import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]

def find_all(a_str, sub):
    start = 0
    ans = []
    while True:
        start = a_str.find(sub, start)
        if start == -1: return ans
        ans.append(start)
        start += len(sub) # use start += 1 to find overlapping matches

def find_token_range(tokenizer, prompt, substring):
    token_array = tokenizer(prompt)['input_ids']
    toks = decode_tokens(tokenizer, token_array)
    # print(toks)
    whole_string = "".join(toks)
    char_locs = find_all(whole_string, substring)
    loc = 0
    cur=0
    tok_start, tok_end = None, None
    possibles = []
    for i, t in enumerate(toks):
        loc += len(t)
        if cur >= len(char_locs):
            break
        if tok_start is None and loc > char_locs[cur]:
            tok_start = i
        if tok_end is None and loc >= char_locs[cur] + len(substring):
            tok_end = i + 1
            possibles.append([tok_start, tok_end])
            tok_start, tok_end = None, None
            cur+=1
    return possibles

device = torch.device("cuda")

num_layers = 32
attn_heads = 32
start_position = 25

def attn_contrib(attn, subject_position, attr_position):
    norms = []
    for i in range(num_layers):
        attn_at_layer = attn[0][i][0]
        norm = torch.max(attn_at_layer[:, subject_position, attr_position]).item()
        print(norm)
        norms.append(norm)
    return norms


'''

num_layers = 32
attn_heads = 32
start_position = 25

import numpy as np
norms = []
for i in range(num_layers):
    attn_at_layer = attn[0][i][0]
    norm = torch.max(attn_at_layer[:, 202, start_position]).item()
    print(norm)
    norms.append(norm)
plt.plot(range(len(norms)), norms)
num_layers = 32
attn_heads = 32
start_position = 25

import numpy as np
norms = []
for i in range(num_layers):
    attn_at_layer = attn[0][i][0]
    norms.append(torch.mean(torch.linalg.vector_norm(attn_at_layer[:, 202, start_position], dim=-1)).item())
num_layers = 32
attn_heads = 32
start_position = 25

import numpy as np

for i in range(num_layers):
    attn_at_layer = attn[0][i][0]
    norms = []
    for pos in range(start_position, length):
        norms.append(torch.mean(torch.linalg.vector_norm(attn_at_layer[:, pos, start_position], dim=-1)).item())
    maxpos = np.argpartition(norms, -4)[-4:] + start_position
    print(i, maxpos, [tokenizer.decode(ids['input_ids'][0][m]) for m in maxpos])
num_layers = 32
attn_heads = 32
start_position = 25

import numpy as np

for i in range(num_layers):
    attn_at_layer = attn[0][i][0]
    norms = []
    for pos in range(start_position, length):
        norms.append(torch.mean(torch.linalg.vector_norm(attn_at_layer[:, pos, start_position], dim=-1)).item())
    maxpos = np.argpartition(norms, -4)[-4:] + start_position
    print(i, maxpos, [tokenizer.decode(ids['input_ids'][0][m]) for m in maxpos])
import numpy as np

for i in range(num_layers):
    attn_at_layer = attn[0][i][0]
    norms = []
    for pos in range(94, 99):
        norms.append(torch.mean(torch.linalg.vector_norm(attn_at_layer[:, pos, 94], dim=-1)).item())
    maxpos = np.argpartition(norms, -4)[-4:] + 94
    print(i, maxpos, [tokenizer.decode(ids['input_ids'][0][m]) for m in maxpos])
sequence_output.sequences[0].shape


'''