import scipy.sparse.linalg as lg
from humanize import naturalsize
from functools import partial
from tqdm.auto import tqdm
from pathlib import Path
import polars as pl
import numpy as np
import logging
import warnings
import umap
import time
import os
import gc
import io
import sys
from config import DATABASE_PATH

lg.lobpcg = partial(lg.lobpcg, verbosityLevel=1)  # Set verbosity level to 1 for lobpcg used in umap
# DATABASE_PATH = Path("/work/msakni2s/database")
warnings.filterwarnings("ignore")

def setup_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

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

def read_data(logger: logging.Logger) -> np.ndarray:

    logger.info("Reading embeddings")
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    embeddings = np.array([], dtype=np.float16)
    
    for i in tqdm(BATCHES, file=tqdm_out):
        df = pl.scan_parquet(i).select("embedding").collect()
        temp = np.vstack(df.to_numpy().flatten(), dtype=np.float16)
        embeddings = np.vstack([embeddings, temp], dtype=np.float16) if embeddings.size else temp
        logger.info(f"Loaded {naturalsize(temp.nbytes)} of embeddings")
        del df
        del temp
        gc.collect()
    logger.info(f"Final embeddings shape: {embeddings.shape}")
    logger.info(f"Total memory used: {naturalsize(embeddings.nbytes)}")

    return embeddings

def convert_time(x):
    return time.strftime("%M:%S", time.gmtime(x))

def run_umap(logger: logging.Logger, embeddings: np.ndarray):
    reducer = umap.UMAP(metric='cosine', verbose=True, unique=True)
    logger.info("Fitting UMAP model")
    reducer.fit(embeddings)

    # logger.info("Saving UMAP model")
    # joblib.dump(reducer, DATABASE_PATH / "umap" / "model.joblib")

    logger.info("Transforming embeddings")
    output = reducer.transform(embeddings)
    return output

DATABASE_PATH_EMBEDDINGS = DATABASE_PATH / "embeddings"
FILES = os.listdir(DATABASE_PATH_EMBEDDINGS)
FILES = [DATABASE_PATH_EMBEDDINGS / chunk for chunk in FILES if chunk.endswith('.parquet')]
BATCHES = []
CHUNK_SIZE = 5 # Polars limitation is 20
for i in range(0, len(FILES), CHUNK_SIZE):
    BATCHES.append([chunk for chunk in FILES[i:CHUNK_SIZE+i]])
    
BATCHES = BATCHES[:1]
if __name__ == "__main__":
    logger = setup_logger()
    start = time.time()
    logger.info(f"Starting Run at {time.ctime()}")

    embeddings = read_data(logger)
    embeddings_2d = run_umap(logger, embeddings)
    ids = pl.scan_parquet(FILES[:5]).select("id").collect().to_numpy().flatten()

    target_path = DATABASE_PATH / "umap" / "umap.parquet"
    logger.info(f"Writing UMAP embeddings to {target_path}") 
    df = pl.DataFrame({
        "id": ids,
        "umap_1": embeddings_2d[:, 0],
        "umap_2": embeddings_2d[:, 1]
    }).write_parquet(target_path)

    end = time.time()
    logger.info(f"Finished Run at {time.ctime()}")
    logger.info(f"Total time taken: {convert_time(end - start)}")