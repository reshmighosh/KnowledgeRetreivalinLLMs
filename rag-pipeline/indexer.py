from ragatouille import RAGPretrainedModel

import json
from tqdm import tqdm 

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
knowns_wiki_documents = json.load(open("knowns_wiki.json"))

RAG.index(
    collection=[k['document'] for k in knowns_wiki_documents], 
    document_ids=[k['entity'] for k in knowns_wiki_documents],
    document_metadatas=[{"entity": "knowns-entity", "source": "wikipedia"} for _ in knowns_wiki_documents],
    index_name="wiki-knowns-1000-full", 
    max_document_length=180, 
    split_documents=True,
)