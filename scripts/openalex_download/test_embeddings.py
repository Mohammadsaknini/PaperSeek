from sentence_transformers.util import cos_sim
from utils import DataReader
from models import Stella
import polars as pl
import numpy as np
import gc
import os

# This script is used to generate the embeddings for the missing works that have been downloaded from
# https://huggingface.co/datasets/colonelwatch/abstracts-embeddings

def get_missing_ids(embeddings_reader: DataReader, metadata_reader: DataReader):
    """
    Check which IDs from openalex that have title and abstract have not yet been embedded.
    """
    embedding_ids = set(embeddings_reader.index_file.keys())
    metadata_ids = set(metadata_reader.index_file.keys())
    missing_ids = metadata_ids - embedding_ids
    # 777_649 missing from the embeddings
    # print(f"Missing IDs: {len(missing_ids)}")

    # 9_098_376 works that have abstract but not title
    # print(f"Missing IDs: {len(embedding_ids - metadata_ids)}")
    return list(missing_ids)

def get_missing_works(missing_ids):
    """
    Create a dataframe that contains the text of the missing works
    """
    metadata_reader = DataReader("E:/openalex-parquet", create_index=False)
    works = []
    for batch in metadata_reader.scan_batches():
        work = pl.scan_parquet(batch).filter(pl.col("id").is_in(missing_ids)).collect()
        works.append(work)
    works = pl.concat(works)
    works = works.with_columns(
        text=(pl.col("title") + " " + pl.col("abstract_inverted_index"))
    ).drop(["title", "abstract_inverted_index"])
    works.write_parquet("missing_works.parquet")
    return works

def generate_test_sample():
    """
    Select a random sample for the orginally generated embedding to be used for comparison with the ones that
    will be generated. We do this to ensure that the downloaded and newly appended embeddings are generated using
    the same techinque and quantiziation.
    """
    path = r"E:\test_sample.parquet"
    if os.path.exists(path):
        df = pl.read_parquet(path)
        return df
    
    df = pl.scan_parquet(r"E:\embeddings\data\data_000.parquet")
    sample = df.slice(0,100).collect()
    ids = sample.select("id").to_numpy().flatten().tolist()
    reader = DataReader(r"E:\openalex-parquet")
    works = reader.get_works(ids)
    works = works.with_columns(
        text=(pl.col("title") + " " + pl.col("abstract_inverted_index"))
    ).drop(["title", "abstract_inverted_index"])
    works = works.join(sample, on="id", how="inner")
    works.write_parquet("test_sample.parquet")
    return works

def generate_embeddings(df: pl.DataFrame):
    model = Stella()
    ids = df["id"].to_list()
    text = df["text"].to_list()
    encodings = model.encode(text, batch_size=128)
    df = pl.DataFrame(
        {
            "id": ids,
            "embedding": encodings,
        }
    )
    df.write_parquet("missing_works_embeddings.parquet")

def generate_missing_works():
    """
    Wrapper function
    """
    path = r"E:\missing_works.parquet"
    if os.path.exists(r"E:\missing_works.parquet"):
        df = pl.read_parquet(path)
        return df
    embeddings_reader = DataReader()
    metadata_reader = DataReader("E:/openalex-parquet")
    missing_ids = get_missing_ids(embeddings_reader, metadata_reader)
    del embeddings_reader, metadata_reader
    gc.collect()
    works = get_missing_works(missing_ids)
    return works

def compare_embeddings():
    df = generate_test_sample()
    original_embeddings = np.vstack(df.select("embedding"))
    model = Stella()
    text = df["text"].to_list()
    new_embeddings = model.encode(text, batch_size=128)
    cos_similarities = np.diag(cos_sim(original_embeddings, new_embeddings))
    # Since there is a lot of factors that can affect the embeddings, such as the inital seed, compression, etc.
    # As long as the cosine similarity is above 0.99, we can consider the embeddings to be the same.
    print(all(cos_similarities> .99))
    return np.diag(cos_similarities)

if __name__ == "__main__":
    df = generate_missing_works()
    generate_embeddings(df)
    # compare_embeddings()
