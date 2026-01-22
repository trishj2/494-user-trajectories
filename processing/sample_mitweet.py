
import polars as pl

_URL = "https://raw.githubusercontent.com/LST1836/MITweet/refs/heads/main/data/MITweet.csv"

# Load MITweet dataset
df = pl.read_csv(_URL, null_values="-1") # Data downloaded from: https://github.com/LST1836/MITweet

# Make sure to store in correct type
df = df.select([
    pl.col("topic"),
    pl.col("tweet"),
    pl.col("tokenized tweet"),
    *[pl.col(col).cast(pl.Int16) for col in df.columns if col.startswith("I")],
    *[pl.col(col).cast(pl.Int16) for col in df.columns if col.startswith("R")],
])

# We'll need this for calculating percentages later
df = df.with_columns(
    num_left=pl.sum_horizontal([(pl.col(col) == 0).cast(pl.Int8) for col in df.columns if col.startswith("I")]),
    num_center=pl.sum_horizontal([(pl.col(col) == 1).cast(pl.Int8) for col in df.columns if col.startswith("I")]),
    num_right=pl.sum_horizontal([(pl.col(col) == 2).cast(pl.Int8) for col in df.columns if col.startswith("I")]),
    num_non_null=pl.sum_horizontal([(pl.col(col).is_not_null()).cast(pl.Int8) for col in df.columns if col.startswith("I")]),
)

# Pcts will be used to determine partisan lean
df = df.with_columns(
    pct_left=(pl.col("num_left") / pl.col("num_non_null")).round(2).fill_null(0.0),
    pct_center=(pl.col("num_center") / pl.col("num_non_null")).round(2).fill_null(0.0),
    pct_right=(pl.col("num_right") / pl.col("num_non_null")).round(2).fill_null(0.0),
)

# If 100% facets are left, right, center, or mixed, assign that label
df = df.with_columns(
    partisan_lean=
        pl.when(pl.col("pct_left") == 1.0).then(pl.lit("LEFT"))
        .when(pl.col("pct_right") == 1.0).then(pl.lit("RIGHT"))
        .when(pl.col("pct_center") == 1.0).then(pl.lit("CENTER"))
        .when(pl.col("pct_left").is_null() & pl.col("pct_center").is_null() & pl.col("pct_right").is_null()).then(pl.lit("NOT POLITICAL"))
        .otherwise(pl.lit("MIXED"))
)

df = df.select("topic", "tweet", "partisan_lean")
df_100 = df.sample(100, seed=195724)

df.write_csv("data/mitweet_sample.csv")