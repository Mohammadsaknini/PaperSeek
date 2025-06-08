from config import DATABASE_PATH
from utils import DataReader
import polars as pl
from tqdm.auto import tqdm

if __name__ == "__main__":
    path = "D:/merged"
    metadata_reader = DataReader(batch_size=5)
    embeddings_reader = DataReader(DATABASE_PATH / "embeddings", batch_size=5)
    for i, embeddings_df in enumerate(embeddings_reader.scan_batches()):
        ids = embeddings_df.select("id").collect().to_numpy().flatten().tolist()
        meta_df = metadata_reader.get_works(ids).lazy()
        df = meta_df.join(embeddings_df, on="id").collect()
        df.write_parquet(path + f"/chunk_{i}.parquet")

