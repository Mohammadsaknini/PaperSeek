from sentence_transformers.util import pytorch_cos_sim
from speedict import Rdict, AccessType, WriteBatch
from typing import Generator
from tqdm.auto import tqdm
from typing import Literal
from models import Stella
from pathlib import Path
from glob import glob
import psutil as ps
import polars as pl
import numpy as np
import torch
import gc


class QueryResponse:
    def __init__(self, ids: list[str], scores: list[float], texts: list[str]):
        self.ids = ids
        self.scores = scores
        self.texts = texts
        self._index = 0

    def __repr__(self) -> str:
        return f"QueryResponse(ids={self.ids}, scores={self.scores})"

    def __getitem__(self, index: int) -> tuple[str, float]:
        return self.ids[index], self.scores[index]

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self.ids):
            raise StopIteration
        result = (self.ids[self._index], self.scores[self._index])
        self._index += 1
        return result


class DataReader:
    def __init__(self, path: str = None, batch_size: int = 1, create_index=False):
        if path is None:
            from config import ROOT_PATH

            self.path = ROOT_PATH
        else:
            if not Path(path).exists():
                raise FileNotFoundError("The path does not exist")

            self.path = Path(path)

        self.files = glob(str(self.path / "*.parquet"))

        if create_index:
            if Path.exists(self.path / "index"):
                self.index_file = self._load_index_file()
            else:
                print("Index file does not exist. Creating one now...")
                self.index_file = self._create_index_file()

        if batch_size is None:
            # check the available memory
            available_memory = ps.virtual_memory().available
            average_file_size = (
                7 * 1024**3
            )  # Average file size is 1.1 GB after being loaded into memory
            num_files = len(self.files)
            batch_size = round(min(available_memory // average_file_size, num_files))
            self.batch_size = batch_size
        else:
            self.batch_size = batch_size

        self.num_batches = len(self.files) // batch_size

    def _get_batches(self):
        return [
            self.files[i : i + self.batch_size]
            for i in range(0, len(self.files), self.batch_size)
        ]

    def read_batches(self) -> Generator[pl.DataFrame, None, None]:
        for batch in tqdm(
            self._get_batches(), total=self.num_batches, desc="Reading batches"
        ):
            yield pl.read_parquet(batch)

    def scan_batches(self) -> Generator[pl.LazyFrame, None, None]:
        for file in tqdm(
            self._get_batches(), total=self.num_batches, desc="Scanning batches"
        ):
            yield pl.scan_parquet(file)

    def _create_index_file(self) -> dict:
        """
        Create an index file for the parquet files which contains a key-value pair of the file name and the works it contains

        In the following format:
        {
            "file_name": ["work_id_1", "work_id_2", ...]
        }

        The index will be stored in a rocksdb database under self.path/index

        Parameters
        ----------
        path: str
            The path to save the index file

        Returns
        -------
        dict
            The index file
        """
        path = str(self.path / "index")
        db = Rdict(str(path))
        for file in tqdm(self.files):
            writer = WriteBatch()
            df = pl.read_parquet(file)
            works = df["id"].to_list()
            for work in works:
                writer.put(work, file)
            db.write(writer)
        db.close()

        return self._load_index_file()

    def _load_index_file(self) -> dict:
        """
        Load the index file

        Parameters
        ----------

        path: str
            The path to the index file

        Returns
        -------
        dict
            The index file
        """
        path = str(self.path / "index")
        db = Rdict(path, access_type=AccessType.read_only())
        return db

    def check_works(self, work_ids: list[str]) -> list[str]:
        """
        Check if the works are in the index file

        Parameters
        ----------
        work_ids: list[str]
            The list of work ids to check

        Returns
        -------
        list[str]
            The list of work ids that are in the index file
        """
        works = []
        for work_id in work_ids:
            try:
                self.index_file[work_id]
                works.append(work_id)
            except KeyError:
                continue
        return works

    def get_work(self, work_id: str) -> pl.DataFrame:
        try:
            file = self.index_file[work_id]
        except KeyError:
            return None

        return pl.scan_parquet(file).filter(pl.col("id") == work_id).collect()

    def get_works(self, work_ids: list[str]) -> pl.DataFrame:
        if isinstance(work_ids, str):
            work_ids = [work_ids]

        files = []
        for work_id in work_ids:
            try:
                files.append(self.index_file[work_id])
            except KeyError:
                print(f"Work {work_id} not found in the index file. Skipping...")
                continue

        files = list(set(files))
        if len(files) == 0:
            return pl.LazyFrame()

        temp_df = None
        if len(files) > self.batch_size:
            for i in tqdm(
                range(0, len(files), self.batch_size), desc="Reading in batches"
            ):
                batch = files[i : i + self.batch_size]
                result = (
                    pl.scan_parquet(batch)
                    .filter(pl.col("id").is_in(work_ids))
                    .collect()
                )
                if temp_df is None:
                    temp_df = result
                else:
                    temp_df = pl.concat([temp_df, result])

            return temp_df
        return pl.scan_parquet(files).filter(pl.col("id").is_in(work_ids)).collect()

    def get_random_sample(self, n: int = 20000) -> pl.DataFrame:
        file = self.files[np.random.randint(0, len(self.files))]
        df = pl.scan_parquet(file).filter(
            pl.col("abstract").is_not_null() & pl.col("title").is_not_null()
        )
        sample = df.collect().sample(n)
        while len(sample) < n:
            file = self.files[np.random.randint(0, len(self.files))]
            df = pl.scan_parquet(file).filter(
                pl.col("abstract").is_not_null()
                & pl.col("title").is_not_null()
                & pl.col("language")
                == "en"
            )
            sample = pl.concat([sample, df.collect().sample(n - len(sample))])
        return sample[["id", "title", "abstract", "embedding"]]

    def _get_checkpoint_row(
        self, df: pl.LazyFrame, last_work_id: str
    ) -> tuple[bool, pl.DataFrame]:
        has_id = not df.filter(pl.col("id") == last_work_id).collect().is_empty()
        if has_id:
            row_idx = (
                df.select("id")
                .with_row_index()
                .filter(pl.col("id") == last_work_id)
                .collect()["index"]
                .item()
            )
            return True, row_idx

        return False, None

    def _sort_results(self, results: list[QueryResponse], n_hits: int):
        for result in results:
            sorted_indices = np.argsort(result.scores)[::-1]
            result.ids = [result.ids[i] for i in sorted_indices][:n_hits]
            result.scores = [result.scores[i] for i in sorted_indices][:n_hits]
            result.texts = [result.texts[i] for i in sorted_indices][:n_hits]

        return results

    def get_corpous(
        self,
        batch_size=200_000,
        last_work: str = "",
    ) -> Generator[tuple[list[str], list[str]], None, None]:
        template = """Title: {title}[SEP]\n 
        Abstract: {abstract}
        """
        checkpoint_found = False
        for i, df in enumerate(self.scan_batches()):
            # Load from checkpoint if given
            if not checkpoint_found and last_work:
                is_checkpoint, row_idx = self._get_checkpoint_row(df, last_work)
                if is_checkpoint:
                    checkpoint_found = True
                    df.slice(row_idx + 1)
                    print(f"Checkpoint found at {self.files[i]}, at row {row_idx}")
                else:
                    continue

            df = df.filter(
                pl.col("abstract").is_not_null()
                & pl.col("title").is_not_null()
                & (pl.col("language") == "en")
            ).collect()

            for i in range(0, len(df), batch_size):
                data = []
                ids = []
                batch = df[i : i + batch_size]
                for row in batch.iter_rows(named=True):
                    ids.append(row["id"])
                    data.append(
                        template.format(title=row["title"], abstract=row["abstract"])
                    )

                yield (ids, data)

    def fetch(
        self,
        query: np.ndarray | list[str],
        n_hits: int = 10_000,
        prompt_name: Literal["s2p_query", "s2s_query"] = "s2p_query",
    ) -> list[QueryResponse]:
        model = Stella()
        if (
            isinstance(query, list) and all(isinstance(q, str) for q in query)
        ) or isinstance(query, str):
            query = model.encode(
                query, convert_to_tensor=True, prompt_name=prompt_name
            ).float()

        # HyDe
        elif isinstance(query, list) and all(isinstance(q, list) for q in query):
            temp = []
            for q in query:
                embeddings: torch.Tensor = model.encode(
                    q,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    prompt_name=prompt_name,
                ).float()
                temp.append(embeddings.mean(dim=0))
            query = torch.stack(temp).float()

        if query.ndim == 1:
            query = query[None, ...]

        # Clean up cause sentence transformer does not release memory
        # model = model.model.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()

        results = []  # type: list[QueryResponse]
        for df in self.scan_batches():
            items = (
                (
                    df.with_columns(
                        text="Title: "
                        + pl.col("title")
                        + "[SEP]\n"
                        + pl.col("abstract")
                    )
                )
                .select("id", "embedding", "text")
                .collect()
                .to_numpy()
            )
            ids = items[:, 0]
            corpus = torch.from_numpy(np.vstack(items[:, 1])).cuda()
            texts = items[:,2]
            scores = pytorch_cos_sim(query, corpus)
            v, idx = torch.topk(scores, n_hits, dim=-1)
            idx = idx.cpu().numpy()
            v = v.cpu().numpy()
            ids = ids[idx]
            texts = texts[idx]
            # Sort the texts df to the same order as the ids
            response = [
                QueryResponse(
                    ids=ids[i].tolist(),
                    scores=v[i].tolist(),
                    texts=texts[i].tolist()
                )
                for i in range(query.shape[0])
            ]
            if len(results) == 0:
                results.extend(response)
            else:
                for i in range(len(results)):
                    results[i].ids.extend(response[i].ids)
                    results[i].scores.extend(response[i].scores)
                    results[i].texts.extend(response[i].texts)

            # Reduce memory footprint
            if len(results[0]) > 100_000:
                results = self._sort_results(results, n_hits)

        # Sort the results by score and return the top_n
        for result in results:
            results = self._sort_results(results, n_hits)

        torch.cuda.empty_cache()
        gc.collect()

        return results
