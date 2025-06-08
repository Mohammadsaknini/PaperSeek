from __future__ import annotations
from sentence_transformers import util
from itertools import batched
from tqdm.auto import tqdm
from torch import device
import pandas as pd
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models import BaseModel

def evaluate_models(query: str, corpus: dict[str: str], 
                    models: list[BaseModel], devices: list[device]) -> pd.DataFrame:
    """
    Evaluate the performance of the models on the given query and corpus.

    Parameters
    ----------

    query : str
        The query to search for in the corpus.

    corpus : dict[str: str]
        A dictionary of corpus items where the key is the ID of the item and the value is the text of the item.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the evaluation. Each row contains the model name, ID, and similarity score.
    """
    chunk_size = 500000
    corpus_ids = list(corpus.keys())
    corpus = list(corpus.values())
    scores = []
    
    from models import BaseModel # type: ignore

    # all the instances of the models must be of type BaseModel else raise an error
    if not all(issubclass(model, BaseModel) for model in models):
        raise ValueError("All the instances of the models must be of type BaseModel")

    results = []
    pbar = tqdm(models)
    for model in pbar:
        name = model.__name__
        pbar.set_description(f"Evaluating {name}")

        model = model(device=devices) # type: BaseModel
        scores = []
        euc_scores = []
        query_embeddings = model.encode_query(query)
        
        # Average the embeddings incase of multiple queries (HyDe Integration)
        if query_embeddings.ndim == 2:
            query_embeddings = query_embeddings.mean(axis=0)
            
        corpus_embeddings = model.encode(corpus, device=devices)
        chunks = batched(corpus_embeddings, chunk_size)
        num_chunks = len(corpus_embeddings) // chunk_size
        pbar2 = tqdm(chunks, total=num_chunks, leave=False)
        for chunk in pbar2:
            pbar2.set_description("Processing chunks")
            score = util.cos_sim(query_embeddings, np.vstack(chunk)).flatten()
            euc_score = util.euclidean_sim(query_embeddings, np.vstack(chunk)).flatten()
            scores.extend(score)
            euc_scores.extend(euc_score)

        for i, score in enumerate(scores):
            results.append({
                "model": name,
                "id": corpus_ids[i],
                "score": score,
                "euc_score": euc_scores[i]
            })

    return pd.DataFrame(results)