import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from attn_contrib import *
import json
from tqdm import tqdm

device = torch.device("cuda")
modelname = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(modelname, 
                                            output_attentions=True, 
                                            cache_dir="/work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/.cache",
                                            return_dict_in_generate=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(modelname, cache_dir="/work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/.cache")

model_helper = Mistral7BHelper(model, tokenizer)

aug_dataset = json.load(open("aug_dataset.json"))
all_norms = []


for i, entry in enumerate(tqdm(aug_dataset)):
    prompt = entry['prompt']
    subject = ''.join(entry['subject'].split())
    attribute = entry['attribute']
    norms = model_helper.decode_all_layers(prompt, subject, attribute)
    all_norms.append(norms)
    plt.clf()
    plt.title("weighted L2 norm of attribute embeddings ")
    plt.xlabel("Layer")
    plt.ylabel("Contribution")
    plt.plot(range(len(norms)), norms)
    sub = '_'.join(subject.split())
    attr = '_'.join(attribute.split())
    figname = sub + "_" + attr
    plt.savefig(f"norm_contribution_plots/{figname}.png")

mean_norm = np.mean(np.array(all_norms), axis=0)
plt.clf()
plt.title("weighted L2 norm of attribute embeddings ")
plt.xlabel("Layer")
plt.ylabel("Contribution")
plt.plot(range(len(mean_norm)), mean_norm)
figname = "average"
plt.savefig(f"norm_contribution_plots/{figname}.png")
