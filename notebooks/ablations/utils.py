import pandas as pd
from typing import Literal

# List of topics with less than 10k publications
TOPICS_UNDER_10K = {
    "Software Process Line": 167,
    "Pharmacokinetics and Associated Efficacy of Emicizumab in Humans": 248,
    "The rodent object-in-context task": 480,
    "Cerebral Small Vessel Disease and the Risk of Dementia": 982,
    "Business Process Meta Models": 1598,
    "Data Stream Processing Latency": 1907,
    "Specialized psychotherapies for adults with borderline personality disorder": 2831,
    "Coronary heart disease, heart failure, and the risk of dementia": 5435,
    "Bayesian PTSD-Trajectory Analysis with Informed Priors": 6395,
    "Cloud Migration": 7909,
}

TOPICS_UNDER_5K = {
    "Software Process Line": 167,
    "Pharmacokinetics and Associated Efficacy of Emicizumab in Humans": 248,
    "The rodent object-in-context task": 480,
    "Cerebral Small Vessel Disease and the Risk of Dementia": 982,
    "Business Process Meta Models": 1598,
    "Data Stream Processing Latency": 1907,
    "Specialized psychotherapies for adults with borderline personality disorder": 2831,
}


def select_core_pub(
    topics: str, type: Literal["worst", "average", "best", "random"]
) -> tuple[list[str], list[str], list[str]]:
    """
    Selects core publications based on the specified type (worst, average, best, random) for the given topics.
    Returns a tuple containing the selected core publication IDs, their corresponding texts, and the topics.
    """
    path = "ablation/cp_effect_s2p"
    texts = []
    core_pubs = []
    core_df = pd.read_excel("data/eval_cps.xlsx")
    for topic in topics:
        df = pd.read_excel(f"{path}/{topic}.xlsx")
        df.sort_values(by="n_cores", ascending=False, inplace=True)

        if type == "average":
            core_id = df.iloc[len(df) // 2]["core_id"]
        elif type == "best":
            core_id = df.iloc[0]["core_id"]
        elif type == "worst":
            core_id = df.iloc[-1]["core_id"]
        elif type == "random":
            core_id = df.sample(1)["core_id"].values[0]
        else:
            raise ValueError("Invalid type")

        core_pub = core_df[core_df["id"] == core_id]
        text = f"Title: {core_pub['title'].iloc[0]}[SEP]\n Abstract: {core_pub['abstract'].iloc[0]}"
        core_pubs.append(core_id)
        texts.append(text)

    return core_pubs, texts, topics
