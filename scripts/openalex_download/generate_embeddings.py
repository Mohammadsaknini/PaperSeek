import sys
sys.path.append(".")
from models import Stella
from pathlib import Path
from tqdm import tqdm
from glob import glob
import polars as pl
import argparse
import platform

if platform.system() == "Windows":
    SOURCE_DIR = r"E:\openalex-parquet"
    TARGET_DIR = r"E:\final"
elif platform.system() == "Linux":
    SOURCE_DIR = r"/work/msakni2s/combined-parquet"
    TARGET_DIR = r"/work/msakni2s/final"
Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)


def run(idx: int, files: list[str],model: Stella, batch_size=128, dry_run=False):
    for file in tqdm(files, desc="Processing files", file=sys.stderr):
        df = pl.read_parquet(file).rename({"abstract_inverted_index": "abstract"})
        df = df.fill_null("").with_columns(
            text="Title: "
            + pl.col("title")
            + "[SEP]\n Abstract: "
            + pl.col("abstract"),
        )
        print(f"Null Values: {df.select("text").null_count()}", file=sys.stderr)
        if dry_run:
            df = df.head(1000)

        embeddings = model.encode(
            df["text"].to_list(),
            show_progress_bar=True,
            batch_size=batch_size,
        )
        df = df.with_columns(pl.lit(embeddings).alias("embedding")).drop("text")

        # 2M rows slices (around 4.5GB per file)
        slice_size = 2_000_000
        slice_idx = 0
        if len(df) > slice_size:
            for i in range(0, len(df), slice_size):
                df.slice(i, slice_size).write_parquet(
                    f"{TARGET_DIR}/chunk_{idx}_{slice_idx}.parquet",
                )
                slice_idx += 1
                
        else:
            df.write_parquet(
                f"{TARGET_DIR}/chunk_{idx}_{slice_idx}.parquet",
            )
        idx += 1


if __name__ == "__main__":
    step_size = 1
    files = glob(f"{SOURCE_DIR}/*.parquet")
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-f", "--file_group",
        type=int,
        choices=[i for i in range(len(files)//step_size)],
        required=True,
        help="Specify the file group"
        )
    argparser.add_argument("-bs", "--batch_size", type=int, default=512)
    argparser.add_argument("-dr", "--dry_run", action="store_true")
    args = argparser.parse_args()
    groups = []
    for i in range(0, len(files), step_size):
        groups.append((i, files[i:i+step_size]))

    idx, group = groups[args.file_group]
    print(f"Index: {idx}, Files:{group}", file=sys.stderr)
    model = Stella()
    run(idx, group, model, batch_size=args.batch_size, dry_run=args.dry_run)
