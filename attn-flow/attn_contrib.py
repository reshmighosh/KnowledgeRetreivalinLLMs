import torch
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


def get_focus_tokens_to(model, tokenizer, prompt, subject, attribute, maxlength=2048, topk=5, device=torch.device("cuda")):
    '''
    
    This method fetches what tokens does a given token pay most attention to at every layer.
    For eg: In a prompt of c1, c2, c3, ... cn, it can fetch the tokens that c_i paid most attention to, 
    at every layer. (which will be amongst c1, c2, ... c_{i})

    If we consider each attention head as a matrix of dimensions (len, len)

    In get_focus_tokens_to - we measure the max across a row.
    In get_focus_tokens_from - we measure max across a column.
    
    '''
    
    ids = tokenizer(prompt, return_tensors='pt').to(device)
    subject_start, subject_end = find_token_range(tokenizer, prompt, subject)[-1]
    sequence_output = model.generate(**ids, max_length=maxlength)
    sequence_length = ids['input_ids'][0].shape[-1]
    generation=tokenizer.decode(sequence_output['sequences'][0][sequence_length:])
    print(generation)
    if attribute not in generation:
        print("Possibly incorrect: model did not answer correctly")
    attn = sequence_output.attentions
    overall_top_k = []
    num_layers=32
    for i in range(num_layers):
        norms = []
        attn_at_layer = attn[0][i][0]
        seq_len = attn_at_layer.shape[-1]
        for pos in range(subject_end):
            norm = torch.max(attn_at_layer[:, subject_end-1, pos])
            norms.append(norm.item())
        top_k = np.argsort(norms)[-topk:][::-1]
        top_k_tokens = []
        for t in top_k:
            top_k_tokens.append(tokenizer.decode(ids['input_ids'][0][t]))
        overall_top_k.append(top_k_tokens)

    appears = -1
    appears_at_top = -1
    for layer, topk in enumerate(overall_top_k):
        # print(f"Layer: {layer}, top tokens: {topk}")
        if attribute in topk and appears < 0:
            appears = layer
        if attribute == topk[0] and appears_at_top < 0:
            appears_at_top = layer
        
        if appears_at_top >= 0 and appears >= 0:
            break
    
    return overall_top_k, appears, appears_at_top


def get_generation(model, tokenizer, prompt, device=torch.device("cuda"), maxlength=1024):
    ids = tokenizer(prompt, return_tensors='pt').to(device)
    sequence_length = ids['input_ids'][0].shape[-1]
    sequence_output = model.generate(**ids, max_length=maxlength)
    generation=tokenizer.decode(sequence_output['sequences'][0][sequence_length:])
    return generation

def get_focus_tokens_from(model, tokenizer, prompt, subject, attribute, maxlength=1024, topk=5, device=torch.device("cuda")):
    '''
    
    This method fetches what tokens paid the most attention to a given token.
    For eg: In a prompt of c1, c2, c3, ... cn, it can fetch the tokens that paid the most attention to c_i. This would be all tokens
    from c_{i} to c_n.

    If we consider each attention head as a matrix of dimensions (len, len)

    In get_focus_tokens_to - we measure the max across a row.
    In get_focus_tokens_from - we measure max across a column.


    '''
    
    ids = tokenizer(prompt, return_tensors='pt').to(device)
    subject_start, subject_end = find_token_range(tokenizer, prompt, subject)[0]
    sequence_output = model.generate(**ids, max_length=maxlength)
    sequence_length = ids['input_ids'][0].shape[-1]
    generation=tokenizer.decode(sequence_output['sequences'][0][sequence_length:])
    print(generation)
    if attribute not in generation:
        print("Possibly incorrect: model did not answer correctly")
    attn = sequence_output.attentions
    overall_top_k = []
    overall_top_k_idx = []
    overall_top_k_positions = []
    num_layers=32
    for i in range(num_layers):
        norms = []
        attn_at_layer = attn[0][i][0]
        seq_len = attn_at_layer.shape[-1]
        for pos in range(subject_start, seq_len):
            norm = torch.max(attn_at_layer[:, pos, subject_start])
            norms.append(norm.item())
        top_k = np.argsort(norms)[-topk:][::-1] + subject_start
        top_k_idx = []
        top_k_tokens = []
        for t in top_k:
            top_k_tokens.append(tokenizer.decode(ids['input_ids'][0][t]))
            top_k_idx.append(ids['input_ids'][0][t])
        overall_top_k.append(top_k_tokens)
        overall_top_k_idx.append(top_k_idx)
        overall_top_k_positions.append(top_k)

    for layer, (topk, topk_idx, positions) in enumerate(zip(overall_top_k, overall_top_k_idx, overall_top_k_positions)):
        print(f"Layer: {layer}, top tokens: {topk}")


class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,)+output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None
    

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None


    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))
        attn_output = self.block.self_attn.activations
        self.attn_mech_output_unembedded = self.unembed_matrix(self.norm(attn_output))
        attn_output += args[0]
        self.intermediate_res_unembedded = self.unembed_matrix(self.norm(attn_output))
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_output_unembedded = self.unembed_matrix(self.norm(mlp_output))
        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()

    def get_attn_activations(self):
        return self.block.self_attn.activations
    

class Mistral7BHelper:
    def __init__(self, model, tokenizer, device=torch.device("cuda")):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, self.model.lm_head, self.model.model.norm)

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_length=max_length)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
          sequence_output = self.model(inputs.input_ids.to(self.device))
          logits = sequence_output.logits
          return logits
        
    def get_attention_probabilities(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
          sequence_output = self.model(inputs.input_ids.to(self.device))
          attentions = sequence_output.attentions
          return attentions

    def set_add_attn_output(self, layer, add_output):
        self.model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        return self.model.model.layers[layer].get_attn_activations()

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, 10)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(label, list(zip(tokens, probs_percent)))
    
    def decode_all_layers(self, text, subject, attribute):
        attr_positions = find_token_range(self.tokenizer, text, attribute)
        indexes = []
        for s,e in attr_positions:
            indexes.extend(list(range(s, e)))
        subj_start, subj_end = find_token_range(self.tokenizer, text, subject)[-1]
        attentions = self.get_attention_probabilities(text)
        norms = []
        for i, layer in enumerate(self.model.model.layers):
            attn_activations = layer.block.self_attn.activations
            attn_at_layer = attentions[i][0]
            embeddings = attn_activations[:, indexes, :].squeeze(0)
            probs = attn_at_layer[:, -1, indexes].squeeze(0).unsqueeze(-1)
            weighted_embeddings = probs * embeddings
            norm = torch.sum(torch.linalg.vector_norm(weighted_embeddings, dim=1))
            norms.append(norm.item())
        return norms

