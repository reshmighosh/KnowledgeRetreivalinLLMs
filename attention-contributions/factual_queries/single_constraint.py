import pickle
import numpy as np
import json

def load_basketball_players():
    with open("./factual_queries/data/basketball_players.pkl", "rb") as f:
        items = pickle.load(f)

    prompt_template = "Tell me the year the basketball player {} was born in."
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: The player was born in"
    for item in items:
        item["constraint"] = item["player_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["constraint"]))
        item["label"] = item["birth_year"]
        item["popularity"] = item["popularity"]
    return items

def load_knowns_1000_no_rag():
    with open("/work/pi_dhruveshpate_umass_edu/rseetharaman_umass_edu/repo-for-paper/attention-contributions/datasets/known_1000.json") as f:
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
        item['prompt'] = f"""You are an expert reasoning bot and identify as "reasoning agent"
You are shown a "user query", and your goal is to complete the missing information in the query by strictly utilizing the information you have learnt and provided as additional context.
Note that the "additional context" will have 5 segments of additional information that you can utilize for completing the query.
Also you are expected to respond to the "user query" without any commentary. Please just respond with the missing information, which is usually one word. Make sure that you abide to this instruction strictly.
Given the "additional context" here: \n {rag_context}, \n please complete the following "user query":\n{item['user_query']}
"""
        item['popularity'] = 1.0
        item['constraint'] = item['object']
    return items

def load_knowns_1000_synthetic_counterfactual(filename="./RAGcontextsynthetic_test50.json"):
    with open(filename) as f:
        items = json.load(f)
    for item in items:
        rag_context = "\n".join(item['context'])
        item['prompt'] = f"""You are an expert reasoning bot and identify as "reasoning agent"
You are shown a "user query", and your goal is to complete the missing information in the query by strictly utilizing the information you have learnt and provided as additional context.
Note that the "additional context" will have 5 segments of additional information that you can utilize for completing the query.
Also you are expected to respond to the "user query" without any commentary. Please just respond with the missing information, which is usually one word. Make sure that you abide to this instruction strictly.
Given the "additional context" here: \n {rag_context}, \n please complete the following "user query":\n{item['user_query']}
"""
        item['popularity'] = 1.0
        item['constraint'] = item['counterfactual_object']
    return items


def load_football_teams():
    with open("./factual_queries/data/football_teams.pkl", "rb") as f:
        items = pickle.load(f)

    prompt_template = "Tell me the year the football team {} was founded in."
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: The team was founded in"
    for item in items:
        item["constraint"] = item["team_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["constraint"]))
        item["label"] = item["founding_year"]
        item["popularity"] = item["popularity"]
    return items


def load_songs():
    with open("./factual_queries/data/songs.pkl", "rb") as f:
        items = pickle.load(f)

    prompt_template = "Tell me the performer of the song {}"
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: The performer is"
    for item in items:
        item["constraint"] = item["song_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["constraint"]))
        item["label"] = item["artist_name"]
        item["popularity"] = item["popularity"]
    return items


def load_movies():
    with open("./factual_queries/data/movies.pkl", "rb") as f:
        items = pickle.load(f)
    prompt_template = "Tell me the director of the movie {}."
    prompt_fn = lambda prompt: f"User: {prompt}\nAssistant: The director is"
    for item in items:
        item["constraint"] = item["movie_name"]
        item["prompt"] = prompt_fn(prompt_template.format(item["constraint"]))
        item["label"] = item["director_name"]
        print(item['constraint'])
    return items

def load_counterfact_subset(subset):
    filename = f"./factual_queries/data/{subset}.pkl"
    items = pickle.load(open(filename, "rb"))
    return items