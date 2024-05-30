import numpy as np
from .single_constraint import *
from .multi_constraint import *

def load_constraint_dataset(dataset_name, filename='', subsample_count=None):
        
    if dataset_name == "basketball_players":
        items = load_basketball_players()
    elif dataset_name == "football_teams":
        items = load_football_teams()
    elif dataset_name == "songs":
        items = load_songs()
    elif dataset_name == "movies":
        items = load_movies()
    elif dataset_name == "known_1000":
        items = load_knowns_1000_no_rag()
    elif dataset_name == "known_1000_synthetic":
        items = load_knowns_1000_synthetic(filename=filename)
    elif dataset_name == "known_1000_synthetic_counterfactual":
        items = load_knowns_1000_synthetic_counterfactual(filename=filename)
    elif "counterfact_" in dataset_name:
        items = load_counterfact_subset(dataset_name)
    elif dataset_name == "nobel":
        items = load_nobel_city()
    elif dataset_name == "words":
        items = load_word_startend()
    elif dataset_name == "books":
        items = load_books()
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if subsample_count is not None:
        # items = np.random.choice(items, subsample_count, replace=False)
        items = items[:subsample_count]
    return items