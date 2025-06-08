import sys
sys.path.append(".")
from qdrant_client.models import VectorParams, Distance, models
from qdrant_client import AsyncQdrantClient
from typing import Generator
from utils import DataReader
from tqdm.auto import tqdm
from uuid import uuid4
import polars as pl
import asyncio
import warnings

warnings.filterwarnings("ignore")
# https://qdrant.tech/articles/vector-search-resource-optimization/
# https://qdrant.tech/documentation/guides/optimize/
# https://qdrant.tech/articles/memory-consumption/
N_POINTS = 200_000
client = AsyncQdrantClient(url="http://localhost:6333", timeout=6000, prefer_grpc=True)

def create_points(df: pl.DataFrame) -> Generator[models.PointStruct, None, None]:
    payloads = df.select(pl.exclude("embedding")).to_dicts()
    embeddings = df.select("embedding").to_numpy().flatten()
    ids = [str(uuid4()) for _ in range(len(embeddings))]
    for i in range(0, len(ids), N_POINTS):
        point_structs = [
            # https://github.com/qdrant/qdrant-client/discussions/469 skip pydantic validation
            models.PointStruct.model_construct(id=ids[j], vector=embeddings[j].tolist(), payload=payloads[j])
            for j in tqdm(range(i, min(i+N_POINTS, len(ids))), desc="Creating points...", leave=False)
        ]
        yield point_structs

async def upload_points(df: pl.DataFrame) -> None:
    n_total = df.shape[0]
    points = create_points(df)
    batch_size = 2500 # Qdrant limitation

    for struct in tqdm(points, total=n_total//N_POINTS, desc="Reading points...", leave=False):
        tasks = [
            client.upsert(collection_name="OpenAlex", points=struct[i : i + batch_size], wait=False)
            for i in range(0, len(struct), batch_size)
        ]
        await asyncio.gather(*tasks)
        # save the last id to resume from
        last_id = struct[-1].payload["id"]
        with open("checkpoint.log", "w") as f:
            f.write(last_id)

# @retry(stop=stop_after_attempt(10))
async def main():
    collection_name = "OpenAlex"

    if await client.collection_exists(collection_name): 
        await client.get_collection(collection_name)
        # await client.update_collection( # Make sure to run once after fully uploading the collection to enable indexing
        # collection_name="OpenAlex",
        # optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000,
        #                                              default_segment_number=20,
        #                             ),
        # )
    else:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE,
                datatype=models.Datatype.FLOAT16,
                on_disk=True,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0
            ),  # Do not forget to set to 20k after inital upload
            hnsw_config=models.HnswConfigDiff(
                on_disk=True, m=32, ef_construct=128
            ),  # Optimize performance over latency.
        )
    reader = DataReader(batch_size=1)
    checkpoint_found = False
    for i, df in enumerate(reader.scan_batches()):
        df = df.collect()
        checkpoint = open("checkpoint.log", "r").read()
        if not checkpoint_found and checkpoint:
            item = df.select("id").with_row_index().filter(pl.col("id") == checkpoint)
            if item.is_empty():
                continue
            else:
                df = df.slice(item["index"].item() + 1)
                checkpoint_found = True
        await upload_points(df)
        checkpoint = True


if __name__ == "__main__":
    asyncio.run(main())
