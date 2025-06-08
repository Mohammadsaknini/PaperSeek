from pathlib import Path
from glob import glob
import platform

if platform.system() == "Linux":
    ROOT_PATH = Path("/home/msakni2s/data/works")
    DATABASE_PATH = Path("/work/msakni2s/database")
else:
    DATABASE_PATH = Path("D:/database")
    ROOT_PATH = Path("D:/embeddings")

FILES = glob(str(ROOT_PATH / "*.parquet"))

def thesis_data_path(folder:str, file:str):
    parent = Path(r"C:\Users\Moham\Desktop\Master_Thesis\paper\data")
    Path(parent / folder).mkdir(parents=True, exist_ok=True)
    return parent / folder / file