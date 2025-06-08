from sentence_transformers import SentenceTransformer
from qdrant_client import models
from utils import QdrantReader, DataReader
from tqdm.auto import tqdm
from pyalex import Works
from glob import glob
import pandas as pd
import warnings
import pyalex
import json
import os

warnings.filterwarnings("ignore")

pyalex.config.email = os.environ.get("EMAIL")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
queries = json.load(open("data/evaluation_data.json", encoding="utf-8"))  # type: dict
litqeval_topics = list(queries.keys())[:7]
synergy_topics = list(queries.keys())[7:]

def invert_abstract(inv_index):
    if inv_index is not None:
        l_inv = [(w, p) for w, pos in inv_index.items() for p in pos]
        return " ".join(map(lambda x: x[0], sorted(l_inv, key=lambda x: x[1])))

def get_dim_core_pubs(topic: str):
    core_pubs_path = "data/dim_cps.xlsx"
    metadata_path = "data/metadata.xlsx"
    core_pubs = pd.read_excel(core_pubs_path)
    core_pubs = core_pubs[core_pubs["Topic"] == topic]
    metadata = pd.read_excel(metadata_path)
    merged_df = pd.merge(
        core_pubs, metadata, left_on="Pub_id", right_on="id", how="inner"
    )[["id", "title", "abstract"]]
    return merged_df

def get_similar_titles(alex_titles, core_titles):
    alex_encoding = model.encode(alex_titles)
    core_encodings = model.encode(core_titles)
    similarities = core_encodings @ alex_encoding.T
    return similarities

def fetch_slr_queries(eval_df, force=False):
    dfs = []
    if os.path.exists("data/slr_query_results.xlsx") or not force:
        return pd.read_excel("data/slr_query_results.xlsx")
    pbar = tqdm(queries.items(), desc="Fetching...")
    for topic, values in pbar:
        pbar.update(1)
        query = values["query"]
        if not query:
            continue

        pager = (
            Works()
            .search_filter(title_and_abstract=query)
            .filter(has_abstract=True, language="en")
            .select(["id", "title"])
            .paginate(method="page", per_page=200, n_max=10_000)
        )

        results = []
        for page in pager:
            results.extend(page)

        result_df = pd.DataFrame(results)
        cps_id = eval_df[eval_df["topic"] == topic]["id"].tolist()
        result_df["topic"] = topic
        result_df["is_core"] = result_df["id"].isin(cps_id)
        print(f"{topic}: {result_df['is_core'].sum()}")
        dfs.append(result_df)
    result_df = pd.concat(dfs)
    result_df.to_excel(
        "data/data/slr_query_results.xlsx",
        index=False,
        engine_kwargs={"options": {"strings_to_urls": False}},
    )        

def litqeval_cps():
    result_dfs = []
    for topic in tqdm(litqeval_topics, desc="Checking Core Availability..."):
        dim_core_df = get_dim_core_pubs(topic)

        # 15 titles at a time
        results = []
        titles = []
        for idx in range(0, len(dim_core_df), 15):
            dim_core_df["title"] = dim_core_df["title"].str.replace(",", "")
            titles.append(
                " OR ".join(
                    ('"' + dim_core_df["title"].iloc[idx : idx + 15] + '"').tolist()
                )
            )

        for title in titles:
            response = (
                Works()
                .search_filter(title=title)
                .filter(has_abstract=True, language="en")
                .select(["id", "title", "abstract_inverted_index"])
                .get(per_page=200)
            )
            results.extend(response)

        result_df = pd.DataFrame(results)
        result_df = result_df[
            ~result_df["title"].str.strip().str.lower().duplicated(keep="first")
        ]
        result_df["abstract"] = result_df["abstract_inverted_index"].apply(
            lambda x: invert_abstract(x)
        )
        result_df.drop("abstract_inverted_index", inplace=True, axis=1)
        pattern = r"[^\w\s]"
        alex_titles = (
            result_df["title"].str.strip().replace(pattern, "", regex=True).tolist()
        )
        core_titles = (
            dim_core_df["title"].str.strip().replace(pattern, "", regex=True).tolist()
        )

        sims = get_similar_titles(core_titles, alex_titles)

        # proof by hand
        # best_fit = sims.argmax(axis=1)
        # for i, idx in enumerate(best_fit):
        #     score = sims[i, idx]
        #     if score < 0.995:
        #         continue
        #     print("----" * 20)
        #     print(score)
        #     print(core_titles[idx])
        #     print(alex_titles[i])

        result_df["score"] = sims.max(axis=1)
        result_df = result_df[result_df["score"] > 0.995]
        result_df["topic"] = topic
        result_dfs.append(result_df)

    dim_core_df = pd.read_excel("data/dim_cps.xlsx")
    openalex_core_df = pd.concat(result_dfs)
    openalex_core_df.drop("score", inplace=True, axis=1)
    openalex_core_df = openalex_core_df[["id", "topic", "title", "abstract"]]
    openalex_core_df.to_excel("data/litqeval_cps.xlsx", index=False)


    return openalex_core_df

def synergy_core_df():
    files = glob("synergy_dataset/*.csv")
    
    # Manually removed
    # Closed source [Muthu_2021]
    # [Radjenovic_2013] Already included in dataset 1
    # [Walker_2018] Dataset says it has 762 CPS, publicaiton says 257

    dfs = []
    for file, topic in zip(files, synergy_topics):
        df = pd.read_csv(file)
        df = df[df["label_included"] == 1]
        df["topic"] = topic
        dfs.append(df)
    synergy_df = pd.concat(dfs)

    final_df = []
    for topic in tqdm(synergy_topics, desc="Checking Core Availability..."):

        cps = synergy_df[synergy_df["topic"] == topic]
        ids = cps["id"].dropna().tolist()
        actual_count = len(ids)
        chunk_size = 20
        if len(ids) > chunk_size:
            works = []
            for i in range(0, len(ids), chunk_size):
                works.extend(Works()[ids[i : i + chunk_size]])
        else:
            works = Works()[ids]
        result_df = pd.DataFrame(works)
        found_count = len(works)
        result_df = result_df[
            result_df["abstract_inverted_index"].notna() & result_df["title"].notna()
        ]
        result_df["abstract"] = result_df["abstract_inverted_index"].apply(
            lambda x: invert_abstract(x)
        )
        filtered_count = len(result_df)
        print(f"{found_count}/{actual_count} -> {filtered_count}")
        result_df["topic"] = topic
        result_df = result_df[["id", "topic", "title", "abstract"]]
        final_df.append(result_df)

    synergy_df = pd.concat(final_df)
    synergy_df.to_excel(
        "data/synergy_cps.xlsx", index=False, engine_kwargs={"options": {"strings_to_urls": False}}
    )
    return synergy_df

def eval_cps(force=False):
    if not os.path.exists("data/litqeval_cps.xlsx") or force:
        litqeval_df = litqeval_cps()
    else:
        litqeval_df = pd.read_excel("data/litqeval_cps.xlsx")

    if not os.path.exists("data/synergy_cps.xlsx") or force:
        synergy_df = synergy_core_df()
    else:
        synergy_df = pd.read_excel("data/synergy_cps.xlsx")

    if not os.path.exists("data/eval_cps.xlsx") or force:
        eval_df = pd.concat([litqeval_df, synergy_df])
        eval_df.to_excel(
            "data/eval_cps.xlsx",
            index=False,
            engine_kwargs={"options": {"strings_to_urls": False}},
        )
    else:
        eval_df =  pd.read_excel("data/eval_cps.xlsx")

    return eval_df

def availablity_overview():
    files = glob("synergy_dataset/*.csv")
    dfs = []
    for file, topic in zip(files, synergy_topics):
        df = pd.read_csv(file)
        df = df[df["label_included"] == 1]
        df["topic"] = topic
        dfs.append(df)

    synergy_actual = pd.concat(dfs)
    litqeval_actual = pd.read_excel("data/dim_cps.xlsx")
    litqeval_found = pd.read_excel("data/litqeval_cps.xlsx")
    synergy_found = pd.read_excel("data/synergy_cps.xlsx")
    for topic in litqeval_topics:
        actual = litqeval_actual[litqeval_actual["Topic"] == topic].groupby("Topic")["Title"].count().to_dict()
        found = litqeval_found[litqeval_found["topic"] == topic].groupby("topic")["id"].count().to_dict()
        print(f"{topic}: {found[topic]}/{actual[topic]}")

    for topic in synergy_topics:
        actual = synergy_actual[synergy_actual["topic"] == topic].groupby("topic")["id"].count().to_dict()
        found = synergy_found[synergy_found["topic"] == topic].groupby("topic")["id"].count().to_dict()
        print(f"{topic}: {found[topic]}/{actual[topic]}")

def validate_in_snapshot_qdrant(df):
    all_ids = df["id"].dropna().tolist()
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
    print(f"Missing IDs: {len(missing_ids)}")
    print(f"Missing IDs: {missing_ids}")

    # remove missing ids from df
    df = df[~df["id"].isin(missing_ids)]
    df.to_excel(
        "data/eval_cps.xlsx",
        index=False,
        engine_kwargs={"options": {"strings_to_urls": False}},
    )

def validate_in_snapshot(df):
    reader = DataReader(batch_size=1)
    all_ids = df["id"].dropna().tolist()
    found = reader.check_works(all_ids)
    missing_ids = set(all_ids).difference(found)
    print(f"Missing IDs: {len(missing_ids)}")
    print(f"Missing IDs: {missing_ids}")
    
if __name__ == "__main__":
    force = True
    # Download and load the validation data
    df = eval_cps(force)
    # Remove the ids that are found in the API but not in the snapshot
    validate_in_snapshot(df) 
    # Run the SLR queries using the API
    fetch_slr_queries(df, force)
    # Check the availability of the core publications (Eval datasets vs OpenAlex)
    availablity_overview()