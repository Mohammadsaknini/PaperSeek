from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
from glob import glob
import polars as pl
import rapidgzip
import msgspec
import warnings
import platform
import gc
import re

warnings.simplefilter(action="ignore", category=FutureWarning)

class Work(msgspec.Struct):
    id: str
    title: str | None
    abstract_inverted_index: dict | None
    language: str | None

if platform.system() == "Windows":
    SOURCE_DIR = r"E:\openalex-snapshot"
    TARGET_DIR = r"E:\openalex-parquet"
    STORAGE_DIR = r"E:\combined-parquet"
    
elif platform.system() == "Linux":
    SOURCE_DIR = r"/work/msakni2s/openalex-snapshot"
    TARGET_DIR = r"/work/msakni2s/openalex-parquet"
    STORAGE_DIR = r"/work/msakni2s/combined-parquet"

Path(TARGET_DIR).parent.mkdir(parents=True, exist_ok=True)

def read_json_lines(paths: str, decoder: msgspec.json.Decoder) -> list[dict]:
    results = {
        "id": [],
        "title": [],
        "abstract_inverted_index": [],
    }
    for path in tqdm(paths, desc="Decompressing files", leave=False):
        with rapidgzip.open(path, parallelization=4) as f:
            data = decoder.decode_lines(f.read())

        # Polars works better with columnar data
        # https://docs.pola.rs/api/python/stable/reference/api/polars.from_records.html

        for i in data:
            if i.language is None or i.abstract_inverted_index is None:
                continue
            if i.language != "en":
                continue
            
            results["id"].append(i.id)
            results["title"].append("" if i.title is None else i.title)
            results["abstract_inverted_index"].append(i.abstract_inverted_index)

    return results

def invert_abstract(inv_index):
    # https://github.com/J535D165/pyalex/blob/main/pyalex/api.py
    if inv_index is not None:
        l_inv = [(w, p) for w, pos in inv_index.items() for p in pos]
        return " ".join(map(lambda x: x[0], sorted(l_inv, key=lambda x: x[1])))

def convert_abstracts(inverted_abstracts: pl.Series):
    abstracts = []
    for i in inverted_abstracts:
        abstracts.append(invert_abstract(i))
    return pl.Series(abstracts, strict=False)

def batch_invert_index(inverted_abstracts: list[pl.Series], batch_size=200_000):
    batches = [inverted_abstracts[i:i+batch_size].to_list() for i in range(0, len(inverted_abstracts), batch_size)]
    results = []
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(convert_abstracts, batch) for batch in batches]
        for future in as_completed(futures):
            results.extend(future.result())
    return pl.Series(results, strict=False)

def filter_and_store():
    decoder = msgspec.json.Decoder(type=Work)
    files = glob(SOURCE_DIR + "/**/*.gz")
    pbar = tqdm(files, desc="Processing files", leave=True)
    chunk_size = 10 # 10 files at a time
    for i in range(0, len(files), chunk_size):
        pbar.set_description(rf"Processing {files[i:i + chunk_size][-1]}")
        data = read_json_lines(files[i:i + chunk_size], decoder)
        if len(data) == 0:
            continue
        df = (
            pl.LazyFrame(
                data=data,
                schema_overrides={
                    "id": pl.Utf8,
                    "title": pl.Utf8,
                    "abstract_inverted_index": pl.Object,
                },
            ).lazy()
            .with_columns( # restore the abstract
                pl.col("abstract_inverted_index")
                .map_batches(convert_abstracts)
                .alias("abstract_inverted_index")
            )
        ).collect()
        if df.height > 0:
            df.write_parquet(TARGET_DIR + f"/chunk_{i // chunk_size}")
        del df
        del data 
        gc.collect()
        pbar.update(chunk_size)

def group_files(file_paths):
    """
    Groups files into bins where each bin's total size is at most 2.5GB.
    
    Uses a first fit decreasing algorithm:
    1. Get each file's size.
    2. Sort files in descending order of size.
    3. Place each file into the first group with enough remaining capacity, or create a new group if necessary.
    
    Args:
        file_paths (list of str): List of file paths.
    
    Returns:
        list of lists: Each inner list contains file paths grouped together.
    """
    # Define the group capacity as 2.5 GB in bytes.
    capacity = int(2.5 * 1024 * 1024 * 1024)
    
    # Create a list of (file_path, size) tuples.
    files_with_sizes = []
    for path in file_paths:
        try:
            size = Path(path).stat().st_size
            files_with_sizes.append((path, size))
        except OSError as e:
            print(f"Warning: Could not access {path}: {e}")
    
    # Sort the files in descending order of size.
    files_with_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # List to store groups; each group is a tuple (remaining_capacity, [file_paths])
    groups = []
    
    for file_path, size in files_with_sizes:
        placed = False
        # Try to fit the file into an existing group.
        for i, (remaining, group) in enumerate(groups):
            if size <= remaining:
                group.append(file_path)
                groups[i] = (remaining - size, group)
                placed = True
                break
        # If it doesn't fit in any group, start a new one.
        if not placed:
            groups.append((capacity - size, [file_path]))
    
    return [group for remaining, group in groups]

def combine_parquet():
    # Store in roughly 2.5GB chunks
    files = glob(TARGET_DIR + "/*")
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    groups = group_files(files)
    for i, group in tqdm(enumerate(groups)):
        df = pl.scan_parquet(group).collect()
        df.write_parquet(f"{STORAGE_DIR}/chunk_{i}.parquet")

if __name__ == "__main__":
    filter_and_store() # Decompress, filter and store the files
    combine_parquet() # Combine the parquet files to 2.5GB chunks