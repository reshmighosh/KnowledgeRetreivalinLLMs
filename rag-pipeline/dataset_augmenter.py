from ragatouille.utils import get_wikipedia_page
import json
from tqdm import tqdm

knowns = json.load(open("known_1000.json"))
subjects = [k['subject'] for k in knowns]
knowns_wiki_documents = []
for s in tqdm(subjects):
    wiki_entity = s.replace(" ", "_")
    knowns_wiki_documents.append({"entity": wiki_entity, "document":get_wikipedia_page(wiki_entity)})

seen = set()
knowns_dedup = []
for k in tqdm(knowns_wiki_documents):
    if k['entity'] in seen:
        continue
    if k['document'] == '' or k['entity'] == None or k['document'] == None:
        continue
    seen.add(k['entity'])
    knowns_dedup.append(k)

json.dump(knowns_dedup, "docs.json")