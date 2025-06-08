import sys
import platform
import os
if platform.system() == "Linux":
    os.environ["HF_HOME"] = "/work/msakni2s/models"
    sys.path.insert(0, "/home/msakni2s/PaperSeek/")
    queries_path = "/home/msakni2s/PaperSeek/data/evaluation_data.json"
    core_pubs_path = "/home/msakni2s/PaperSeek/data/dim_cps.xlsx"
    metadata_path = "/home/msakni2s/PaperSeek/data/metadata.xlsx"
    predicted_pubs_base_path = "/home/msakni2s/PaperSeek/scripts/benchmark_embedding_models/input/"
    output_path = "/home/msakni2s/PaperSeek/scripts/benchmark_embedding_models/output/stella"
else:
    queries_path = "data/evaluation_data.json"
    core_pubs_path = "data/dim_cps.xlsx"
    metadata_path = "data/metadata.xlsx"
    predicted_pubs_base_path = "scripts/benchmark_embedding_models/input/"
    output_path = "scripts/benchmark_embedding_models/output/stella"


from models import (
    SciBert,
    Specter2,
    Specter2Adhoc,
    E5small,
    MiniLm,
    Linq,
    Qwen2,
    Stella,
    evaluate_models,
)
from utils import DataReader, Query
from pathlib import Path
from torch import device
import pandas as pd
import argparse
import json


def get_query_and_corpus(topic: str, test_run=False) -> tuple[str, dict]:
    predicted_pubs_path = f"{predicted_pubs_base_path}{topic.replace(' ', '_')}.csv"
    reader = DataReader()
    core_pubs = pd.read_excel(core_pubs_path)
    core_pubs = core_pubs[core_pubs["Topic"] == topic]
    metadata = pd.read_excel(metadata_path)
    similar_pubs = pd.read_csv(predicted_pubs_path)

    # The core publications
    merged_df = pd.merge(core_pubs, metadata, left_on="Pub_id", right_on="id", how="inner")[["id", "title", "abstract"]]
    
    # A random sample of publications for noise
    sample = reader.get_random_sample().to_pandas()

    data = pd.concat([merged_df, sample, similar_pubs], axis=0).sample(frac=1).reset_index(drop=True)
    data.drop_duplicates(subset=["id"], inplace=True)

    template = """Title: {title}[SEP]\n 
    Abstract: {abstract}
    """
    corpus = {}
    query = Query(topic).format(1)
    rows = data[:100].iterrows() if test_run else data.iterrows()
    for index, row in rows:
        corpus[row["id"]] = template.format(title=row["title"], abstract=row["abstract"]) 
    print(f"Selected Topic: {topic}, Size: {len(data)}")  

    return query, corpus 

def store_results(results: pd.DataFrame, core_pubs: pd.DataFrame, topic: str) -> pd.DataFrame:
    df = results.copy()
    df["score"] = df["score"].apply(lambda x: x.item())
    df["euc_score"] = df["euc_score"].apply(lambda x: x.item())
    df["type"] = df["id"].isin(core_pubs["Pub_id"].values)
    df["core"] = df["type"].apply(lambda x: "Core" if x else "Non-Core")
    df["topic"] = [topic] * len(df)
    df = df[["id", "model", "score", "euc_score", "core"]]
    Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{output_path}/{topic.replace(" ", "_")}_results.csv", index=False)

def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the performance of different embedding models")
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Run a test with a smaller dataset",
    )
    parser.add_argument(
        "-g",
        "--gpu_idx",
        type=int,
        nargs="*",
        default=[0],
        help="GPU indices to use, it can be either 1 or many as follows -g 0 1 2. default is 0",
    )
    parser.add_argument(
        "-i",
        "--item",
        type=int,
        default=0,
        help="The topic index of the topic ot be embedded",
    )
    return parser.parse_args()

def run(test_run: bool, gpu_indices: list[int] = [0], item_idx: int = 0) -> None:
    topics = list(json.load(open(queries_path, encoding="utf-8")).keys())[item_idx]
    topics = [topics]
    models = [Stella]
    core_pubs = pd.read_excel(core_pubs_path)
    devices = [device(type="cuda", index=gpu_idx) for gpu_idx in gpu_indices][0]
    os.environ["CUDA_VISIBLE_DEVICES"]= f"{gpu_indices}"
    for topic in topics:
        print(f"Running for topic: {topic}")
        query, corpus = get_query_and_corpus(topic, test_run=test_run)
        results = evaluate_models(query, corpus, models, devices)
        temp = core_pubs[core_pubs["Topic"] == topic].copy()
        store_results(results, temp, topic)


if __name__ == "__main__":
    cli_args = cli()
    run(cli_args.test, cli_args.gpu_idx, cli_args.item)
