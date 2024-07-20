import pickle
import numpy as np
import json

def load_knowns_1000_no_rag():
    with open("/work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/repo-for-paper/attention-contributions-llama/datasets/known_1000.json") as f:
        items = json.load(f)
    for item in items:
        item['popularity'] = 1.0
        item['constraint'] = item['subject']
    return items

def load_knowns_1000_synthetic(filename="./RAGcontextsynthetic_test50.json"):
    with open(filename) as f:
        items = json.load(f)
    for item in items:
        rag_context = "\n".join(item['context'])
        item['prompt'] = f"""Information is below:----------------
{rag_context}
Given the context information and not prior knowledge, complete the following
\n{item['user_query']}"""
        item['popularity'] = 1.0
        item['constraint'] = item['object']
    return items

def load_knowns_1000_synthetic_counterfactual(filename="./RAGcontextsynthetic_test50.json"):
    with open(filename) as f:
        items = json.load(f)
    for item in items:
        rag_context = "\n".join(item['context'])
        item['prompt'] = f"""Information is below:---------------- 
{rag_context}
Given the context information and not prior knowledge, complete the following
\n{item['user_query']}"""
        item['popularity'] = 1.0
        item['constraint'] = item['counterfactual_object']
    return items

def load_counterfact_subset(subset):
    filename = f"./factual_queries/data/{subset}.pkl"
    items = pickle.load(open(filename, "rb"))
    return items