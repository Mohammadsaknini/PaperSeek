import sys
import platform

if platform.system() == "Linux":
    sys.path.insert(0, "/home/msakni2s/PaperSeek/")
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from config import DATABASE_PATH
from utils import DataReader
from models import Specter2
from tqdm.auto import tqdm
import polars as pl
import argparse
import logging
import torch
import io
import gc
import os
import re

class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)

def setup_logger(filename) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{filename}.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def get_last_chunk(path: str) -> tuple[str, int]:
    files = os.listdir(path)
    files = [file for file in files if file.endswith(".parquet")]
    files = sorted(files, key=lambda x: float(re.findall(r"(\d+)", x)[0]))
    last_file = files[-1]
    chunk_idx = int(re.findall(r"(\d+)", last_file)[0])
    return last_file, chunk_idx

def read_last_work(path: str) -> str:
    last_file, _ = get_last_chunk(path)
    df = pl.scan_parquet(os.path.join(path, last_file))
    last_work = df.select("id").last().collect().item()
    return last_work

def run(
    path,
    corpus_batch_size=200_000,
    encoder_batch_size=32,
    gpu_indices: list[int] = [0],
    load_checkpoint: bool = False,
):
    
    logger = setup_logger("db")
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    logger.info(f"Creating a database at {path}")
    reader = DataReader(batch_size=1, create_index=False)
    model = Specter2()
    total_docs = 88_826_402  # ~88.8M
    num_iterations = total_docs // corpus_batch_size
    devices = [torch.device(type="cuda", index=gpu_idx) for gpu_idx in gpu_indices]
    
    last_work = ""
    chunk_idx = 0
    if load_checkpoint:
        last_chunk, chunk_idx = get_last_chunk(path)
        chunk_idx += 1 # Increment so that we don't overwrite the last chunk
        last_work = read_last_work(path)
        logger.info(
            f"Loading from the last checkpoint at {path}/{last_chunk} and Workid of {last_work}"
        )

    pbar = tqdm(
        reader.get_corpous(corpus_batch_size, last_work),
        mininterval=30,
        total=num_iterations - chunk_idx,
        file=tqdm_out,
    )

    for i, (ids, text) in enumerate(pbar):
        i = i + chunk_idx
        pbar.set_description(f"Processing {len(ids)} documents")
        encodings = model.encode_parallel(
            text, batch_size=encoder_batch_size, devices=devices
        )
        df = pl.DataFrame(
            {
                "id": ids,
                "embedding": encodings,
            }
        )
        df.write_parquet(f"{path}/chunk_{i}.parquet")
        logger.info(f"Saved {path}/chunk_{i}.parquet")
        del encodings
        del df
        torch.cuda.empty_cache()
        gc.collect()

def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a database of embeddings")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default=DATABASE_PATH,
        help="Storage path",
    )
    parser.add_argument(
        "-ebs",
        "--encoder_batch_size",
        type=int,
        default=32,
        help="Batch size for the encoder. Default is 32",
    )
    parser.add_argument(
        "-cbs",
        "--corpus_batch_size",
        type=int,
        default=200_000,
        help="Batch size for the corpus. Default is 200000",
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
        "-lc",
        "--load_checkpoint",
        action="store_true",
        help="Load from the last checkpoint",
    )

    return parser.parse_args()

if __name__ == "__main__":
    cli_args = cli()
    run(
        cli_args.path,
        corpus_batch_size=cli_args.corpus_batch_size,
        encoder_batch_size=cli_args.encoder_batch_size,
        gpu_indices=cli_args.gpu_idx,
        load_checkpoint=cli_args.load_checkpoint,
    )
