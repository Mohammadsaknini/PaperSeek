from utils import QdrantReader, Query, DataReader
import pandas as pd
eval_df = pd.read_excel("data/eval_cps.xlsx")
topics = eval_df["topic"].unique().tolist()
queries = [Query(topic).format(1) for topic in topics]

def run_qdrant():
    reader = QdrantReader()
    results = {"id": [], "score": [], "topic": []}
    for query in queries:
        hits = reader.fetch(query)
        for i, hit in enumerate(hits):
            for point in hit.points:
                results["id"].append(point.payload["id"])
                results["score"].append(point.score)
                results["topic"].append(topics[i])

    df = pd.DataFrame(results)
    df.to_excel("ablation/qdrant_benchmark/qdrant_results.xlsx", index=False, engine_kwargs={"options": {"strings_to_urls": False}})

def full_scan():
    reader = DataReader()
    results = {"id": [], "score": [], "topic": []}
    hits = reader.fetch(queries)
    for i, hit in enumerate(hits):
        results["id"].extend(hit.ids)
        results["score"].extend(hit.scores)
        results["topic"].extend([topics[i]] * len(hit.ids))

    df = pd.DataFrame(results)
    df.to_excel("ablation/qdrant_benchmark/full_scan_results.xlsx", index=False, engine_kwargs={"options": {"strings_to_urls": False}})

def compare_results():
    qdrant_df = pd.read_excel("ablation/qdrant_benchmark/qdrant_results.xlsx")
    full_scan_df = pd.read_excel("ablation/qdrant_benchmark/full_scan_results.xlsx")
    results = {"Qdrant Recall": [], "Full Scan Recall": [], "Intersection": [], "Topic": []}
    for topic in topics:
        cps = eval_df[eval_df["topic"] == topic]["id"].values.tolist()
        qdrant_ids = set(qdrant_df[qdrant_df["topic"] == topic]["id"])
        full_scan_ids = set(full_scan_df[full_scan_df["topic"] == topic]["id"])
        intersection = qdrant_ids.intersection(full_scan_ids)
        qdrant_topic = qdrant_df[qdrant_df["topic"] == topic]
        full_scan_topic = full_scan_df[full_scan_df["topic"] == topic]

        qdrant_recall = qdrant_topic["id"].isin(cps).sum() / len(cps)
        full_scan_recall = full_scan_topic["id"].isin(cps).sum() / len(cps)

        results["Topic"].append(topic)
        results["Qdrant Recall"].append(qdrant_recall)
        results["Full Scan Recall"].append(full_scan_recall)
        results["Intersection"].append(len(intersection))

    df = pd.DataFrame(results, index=topics)
    print("Average Qdrant Recall:", df["Qdrant Recall"].mean())
    print("Average Full Scan Recall:", df["Full Scan Recall"].mean())
    print("Average Intersection:", df["Intersection"].mean())

if __name__ == "__main__":
    # run_qdrant()
    # full_scan()
    compare_results()
        
