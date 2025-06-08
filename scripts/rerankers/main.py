import sys
sys.path.append(".")
from models import (
    BGEReranker,
    MiniLmCE,
    ReciprocalRankFusion,
    BM25Reranker,
    Qwen2,
    BaseReranker,
    BaseModel,
)
from sentence_transformers import util
from utils import HyResearch
from pathlib import Path
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import argparse
import platform
import sys


if platform.system() == "Windows":
    OUTPUT_PATH = "/scripts/rerankers/output/"
elif platform.system() == "Linux":
    OUTPUT_PATH = "/work/msakni2s/PaperSeek/scripts/rerankers/output/"
    
# We only care about the topics that have a query and have under 10k
TOPICS = {
    "Software Process Line": 167,
    "Data Stream Processing Latency": 1907,
    "Business Process Meta Models": 1598,
    "Cloud Migration": 7909,
    "Cerebral Small Vessel Disease and the Risk of Dementia": 982,
    "Pharmacokinetics and Associated Efficacy of Emicizumab in Humans": 248,
    "Specialized psychotherapies for adults with borderline personality disorder": 2831,
    "The rodent object-in-context task": 480,
    "Bayesian PTSD-Trajectory Analysis with Informed Priors": 6395,
    "Coronary heart disease, heart failure, and the risk of dementia": 5435,
}

def get_topic_data(df: pd.DataFrame, topic: str) -> list[str, dict[str, str]]:
    topic_df = df[df["topic"] == topic]
    query = topic_df["query"].iloc[0]
    query = "\n".join(query) if isinstance(query, (list, np.ndarray)) else query
    documents = dict(zip(topic_df["id"].values, [i for i in topic_df["text"].values]))
    return query, documents

def bm25_query(topic: str, query: str) -> str:
    return "\n".join(HyResearch().generate_n_queries(query, topic, 1))

def rerank_base_model(model: BaseModel, query: str, documents: dict[str, str], top_n: int):
    query_embeddings = model.encode_query(query)
    corpus_embeddings = model.encode(list(documents.values()), batch_size=2)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_n)
    indicies = [hit["corpus_id"] for hit in hits[0]]
    documents = list(documents.items())
    results = {documents[i][0]: documents[i][1] for i in indicies}
    del corpus_embeddings
    del query_embeddings

    return results

def rerank(df: pd.DataFrame, rerankers: list[BaseReranker | BaseModel]) -> pd.DataFrame:
    # Ensure ReporicalRankFusion is the last reranker
    if ReciprocalRankFusion in rerankers and rerankers[-1] != ReciprocalRankFusion:
        rerankers.remove(ReciprocalRankFusion)
        rerankers.append(ReciprocalRankFusion)

    results = {
        "Topic": []
    }
    for reranker in rerankers:
        results[reranker.__class__.__name__] =  []
    topics_pbar = tqdm(TOPICS.keys())
    for topic in topics_pbar:
        topics_pbar.set_description(topic)

        topic_results = [] # contians the results of the rerankers to be used in the RRF
        query, documents = get_topic_data(df, topic)
        top_n = TOPICS[topic]
        results["Topic"].extend([topic] * top_n)

        rerankers_pbar = tqdm(rerankers, leave=False) 
        for reranker in rerankers_pbar:
            rerankers_pbar.refresh()
            rerankers_pbar.set_description(reranker.__class__.__name__)

            if reranker.__class__.__name__ ==  "ReporicalRankFusion":
                scores = reranker.rerank(topic_results)
            elif isinstance(reranker, BM25Reranker):
                scores = reranker.rerank(bm25_query(topic, query), documents) 
            elif issubclass(reranker.__class__, BaseReranker): 
                scores = reranker.rerank(query, documents)
                    
            elif issubclass(reranker.__class__, BaseModel):
                scores = rerank_base_model(reranker, query, documents, top_n)
            else:
                raise ValueError("Unexpected model")
            
            topic_results.append(scores)
        
        for i, result in enumerate(topic_results):
            result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
            ids = list(result.keys())[:top_n]
            results[rerankers[i].__class__.__name__].extend(ids)

        
    return pd.DataFrame(results)

def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerankers")
    parser.add_argument( # default: is False
        "--hyqe",
        action="store_true",
        help="Use HyQE for reranking",
    )
    parser.add_argument( # files group (1-9)
        "-f",
        "--file-group",
        type=int,
        choices=[i for i in range(1, 10)],
        required=True,
        help="Specify the file group",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run using a portion of the data"
    )

    return parser.parse_args()

if __name__ == "__main__":
    files = glob("scripts/rerankers/input/**/*.parquet")
    rerankers = [
        BM25Reranker(),
        MiniLmCE(),
        BGEReranker(),
        Qwen2(),
        ReciprocalRankFusion(),
    ]
    args = arg_parser()
    file_group = args.file_group
    dry_run =  args.dry_run

    files = files[file_group - 1 : file_group]
    print(f"Processing files: {files}", file=sys.stderr)
    for file in files:
        file_name = "/".join(file.split("/")[3:])
        path = OUTPUT_PATH + file_name
        folder = "/".join(path.split("/")[0:-1])
        Path(folder).mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(file)
        if dry_run:
            df = df.groupby("topic").head(100)
            TOPICS = {k: 100 for k in TOPICS.keys()}
        results = rerank(df, rerankers)
        results.to_parquet(path)
        