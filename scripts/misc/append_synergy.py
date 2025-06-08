from qdrant_client import models
from utils import QdrantReader
from models import Specter2
from glob import glob
import polars as pl
import pandas as pd
import json

# Closed source
# [Muthu_2021] Muthu, S., & Ramakrishnan, E. (2020). Fragility Analysis of Statistically Significant Outcomes of Randomized Control Trials in Spine Surgery. Spine, 46(3), 198–208. https://doi.org/10.1097/brs.0000000000003645
# Already included in dataset 1
# [Radjenovic_2013] Radjenović, D., Heričko, M., Torkar, R., & Živkovič, A. (2013). Software fault prediction metrics: A systematic literature review. Information and Software Technology, 55(8), 1397–1418. https://doi.org/10.1016/j.infsof.2013.02.009

def synergy_df():
    files = glob("synergy_dataset/*.csv")
    df = pl.scan_csv(files).collect()
    all_ids = (
        df.filter(pl.col("abstract").is_not_null())
        .select(["id"])
        .to_numpy()
        .flatten()
        .tolist()
    )

    api = QdrantReader()
    records = api.client.scroll(
        collection_name="OpenAlex",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="id",
                    match=models.MatchAny(any=all_ids),
                ),
            ]
        ),
        limit=len(all_ids),
    )[0]
    avail_ids = [r.payload["id"] for r in records]
    missing_ids = set(all_ids).difference(avail_ids)
    df = (
        df.filter(pl.col("id").is_in(missing_ids))
        .with_columns(
            referenced_works_count=pl.col("referenced_works").map_elements(
                lambda x: len(eval(x))
            ),
            keywords=pl.lit(None),
            topics=pl.lit(None),
            language=pl.lit("en"),
            fwci=pl.lit(0),
            type=pl.when(pl.col("type") == "journal-article")
            .then(pl.lit("article"))
            .otherwise(pl.col("type")),
        )
        .drop(["label_included", "referenced_works"])
    )

    return df

def add_to_search_results():

    old_df = pd.read_excel("data/slr_query_results.xlsx")
    queries = json.load(open("data/synergy_queries.json"))
    to_remove = ["Muthu_2021", "Radjenovic_2013"]
    files = glob("synergy_dataset/*.csv")

    # pop from list if it contains the name
    for name in to_remove:
        files = [f for f in files if name not in f]
        
    results = []
    for file, topic in zip(files, queries.keys()):
        temp = pd.read_csv(file)
        temp["topic"] = topic
        results.append(temp)

    df = pd.concat(results)
    df = df.rename(columns={"label_included": "is_core"})
    df["is_core"] = df["is_core"].astype(bool)
    df = df[["topic", "id", "is_core"]]
    df = pd.concat([old_df, df])
    df.to_excel(r"data/slr_query_results.xlsx", index=False, engine_kwargs={'options': {'strings_to_urls': False}})

def add_to_core_pubs():
    core_df = pd.read_excel("data/eval_cps.xlsx")
    df = pd.read_excel("data/slr_query_results.xlsx")
    to_select = core_df["topic"].unique().tolist()
    df = df[~df["topic"].isin(to_select) & (df["is_core"])]
    df.drop("is_core", inplace=True, axis=1)
    core_df = pd.concat([core_df, df])
    queries = json.load(open("data/synergy_queries.json"))
    to_remove = ["Muthu_2021", "Radjenovic_2013"]
    files = glob("synergy_dataset/*.csv")

    # pop from list if it contains the name
    for name in to_remove:
        files = [f for f in files if name not in f]
        
    results = []
    for file, topic in zip(files, queries.keys()):
        temp = pd.read_csv(file)
        temp["topic"] = topic
        results.append(temp)

    to_merge = pd.concat(results)
    to_merge = to_merge[to_merge["label_included"] == 1]
    df = pd.merge(core_df, to_merge, on="id", how="left")
    df["topic"] = df["topic_x"].fillna(df["topic_y"])
    df["title"] = df["title_x"].fillna(df["title_y"])
    df["abstract"] = df["abstract_x"].fillna(df["abstract_y"])
    df.drop(["topic_x", "topic_y", "title_x", "title_y", "abstract_x", "abstract_y"], axis=1, inplace=True)
    df = df[["id", "topic", "title", "abstract"]]
    df.to_excel("data/eval_cps.xlsx", index=False, engine_kwargs={"options": {"strings_to_urls": False}})

def create_qdrant_df():
    df = synergy_df()
    template = """Title: {title}[SEP]\n 
            Abstract: {abstract}
            """
    texts = [
        template.format(title=title, abstract=abstract)
        for title, abstract in zip(df["title"], df["abstract"])
    ]
    model = Specter2()

    df = df.with_columns(embedding=model.encode(texts))
    df.write_parquet("synergy_dataset/missing.parquet")

if __name__ == "__main__":
    add_to_search_results()
    add_to_core_pubs()
    create_qdrant_df()

