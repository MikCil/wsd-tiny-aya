import re
from rich import print
import polars as pl


class Eval:
    lang: str
    target_word: str
    context: str
    pos: str
    ref: str
    pred: str | None = None
    sbert_score: float | None = None
    bert_score: float | None = None
    bleurt_score: float | None = None


sense_inventory_df = (
    pl.read_excel("./UAW-WSD-18-Corpus/Sense Inventory.xlsx")
    .select(
        [
            pl.col("Count").alias("index"),
            pl.col("Word").alias("target_word"),
            pl.col("POS").alias("pos"),
            pl.col("Sense1").alias("gloss_1"),
            pl.col("Sense2").alias("gloss_2"),
            pl.col("Sense3").alias("gloss_3"),
            pl.col("Sense4").alias("gloss_4"),
            pl.col("Sense5").alias("gloss_5"),
            pl.col("Sense6").alias("gloss_6"),
            pl.col("Sense7").alias("gloss_7"),
            pl.col("Sense8").alias("gloss_8"),
            pl.col("Sense9").alias("gloss_9"),
            pl.col("Sense10").alias("gloss_10"),
            pl.col("Sense 11").alias("gloss_11"),
        ]
    )
    .sort(by="target_word")
)

examples_df = (
    pl.read_excel(
        "UAW-WSD-18-Corpus/Gloss of each word avaialble in sense inventory.xlsx"
    )
    .select(
        [
            pl.col("Count").alias("index"),
            pl.col("Word").alias("target_word"),
            pl.col("Sense1").alias("example_1"),
            pl.col("Sense2").alias("example_2"),
            pl.col("Sense3").alias("example_3"),
            pl.col("Sense4").alias("example_4"),
            pl.col("Sense5").alias("example_5"),
            pl.col("Sense6").alias("example_6"),
            pl.col("Sense7").alias("example_7"),
            pl.col("Sense8").alias("example_8"),
            pl.col("Sense9").alias("example_9"),
            pl.col("Sense10").alias("example_10"),
            pl.col("Sense 11").alias("example_11"),
        ]
    )
    .sort(by="target_word")
)

cols_to_add = examples_df.drop("target_word").drop("index")
combined = sense_inventory_df.hstack(cols_to_add)

cols_to_unpivot = [
    col for col in combined.columns if re.match(r"^(gloss|example)_[0-9]+$", col)
]

long_df = combined.unpivot(
    index=["target_word", "pos", "index"],
    on=cols_to_unpivot,
    variable_name="temp_col",
    value_name="value",
)

df_split = long_df.with_columns(
    [
        pl.col("temp_col").str.split("_").list.get(0).alias("type"),
        pl.col("temp_col").str.split("_").list.get(1).alias("index"),
    ]
)

df_final = (
    df_split.pivot(
        values="value", index=["index", "target_word", "pos"], columns="type"
    )
    .drop("index")
    .drop_nulls(["gloss", "example"])
)

df_final = df_final.rename(
    {
        "gloss": "ref",
        "example": "context",
    }
).with_columns([pl.lit("ur").alias("lang")])

df_final.write_csv("./evals/test-ur/senses.csv")
