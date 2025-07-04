import polars as pl


df = (
    pl.scan_parquet("ablation/final/cp_random_hyde_1_run-1.parquet")
    .filter(pl.col("topic") == "Software Process Line").with_columns(
        pl.col("text").str.split_exact("[SEP]",1).struct.rename_fields(["title", "abstract"]).alias("fields")
        # abstract=pl.col("text").str.split("[SEP]").arr.last(),
    ).unnest("fields")
    .with_columns(
        pl.col("title").str.replace("Title: ", "").str.strip_chars().alias("title"),
    )
    .collect()
)
eval_df = (
    pl.read_excel("data/eval_cps.xlsx")
    .filter(pl.col("topic") == "Software Process Line")
    .select("id")
).to_numpy().flatten()

df = df.with_columns(
    pl.when(pl.col("id").is_in(eval_df))
    .then(pl.lit(1)).otherwise(pl.lit(0)).alias("included")
).rename({"id": "doi"}).select("doi", "title", "abstract", "included")

df.write_excel("test.xlsx")