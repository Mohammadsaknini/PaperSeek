from models import (
    SciBert,
    Specter2,
    Specter2Adhoc,
    Qwen2,
    Linq,
    Stella,
    MiniLm,
    E5small,
    BaseModel,
)
from utils.datareader import DataReader
from IPython.display import display
from humanize import intword
import polars as pl
import pandas as pd
import json


def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


class Stats:
    def __init__(self):
        self.total_rows = 0
        self.title_na_count = 0

    def update(self, total_rows: int, title_na_count: int):
        self.total_rows += total_rows
        self.title_na_count += title_na_count

    def describe(self):
        output = {}
        output["Number of items"] = f"{self.total_rows} ({intword(self.total_rows)})"
        output["Items missing title"] = (
            f"{self.title_na_count} ({intword(self.title_na_count)})"
        )

        # if ipython is used
        if in_notebook():
            display(pd.DataFrame(output, index=["values"]))
        else:
            print(json.dumps(output, indent=4))

    def __str__(self):
        output = {}
        output["total_rows"] = intword(self.total_rows)
        output["title_na_count"] = intword(self.title_na_count)
        return str(json.dumps(output, indent=4))

    def __repr__(self):
        return str(self)


class EDA:
    def __init__(self, df: pl.LazyFrame):
        self.df = df
        self.total_rows = 0
        self.title_na_count = 0

    def run(self):
        result = (
            self.df.select(
                pl.when(pl.col("title") == "")
                .then(1)
                .otherwise(0)
                .sum()
                .alias("title_na_count"),
                pl.len().alias("total_rows"),
            )
            .collect()
            .to_dicts()[0]
        )

        self.total_rows += result["total_rows"]
        self.title_na_count += result["title_na_count"]

    def openalex_stats():
        reader = DataReader(batch_size=3, create_index=False)
        stats = Stats()
        for batch in reader.scan_batches():
            eda = EDA(batch)
            eda.run()
            stats.update(eda.total_rows, eda.title_na_count)
        stats.describe()

    @staticmethod
    def _count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def models_parameters_count(models: list[BaseModel] = None):
        models = [
            SciBert,
            Specter2,
            Specter2Adhoc,
            Qwen2,
            Linq,
            Stella,
            MiniLm,
            E5small,
        ] if models is None else models 
        for model in models:
            name = model.__name__
            model = model().model
            print(f"{name}: {intword(EDA._count_parameters(model))}")
