import polars as pl

_URL = (
    "https://raw.githubusercontent.com/LST1836/MITweet/refs/heads/main/data/MITweet.csv"
)

# Column name mappings based on MITweet paper (EMNLP 2023)
# Schema: 5 Domains -> 12 Facets
# Domains: Politics, Economy, Culture, Diplomacy, Society
# Ideology labels: 0=Left, 1=Center, 2=Right, -1=Unrelated
# Relevance labels: 0=Unrelated, 1=Related
IDEOLOGY_COLS = {
    # Politics Domain
    "I1": "ideology_political_regime",  # Socialism vs Capitalism
    "I2": "ideology_state_structure",  # Federalism vs Unitarism
    # Economy Domain
    "I3": "ideology_economic_orientation",  # Regulation vs Free Market
    "I4": "ideology_economic_equality",  # Egalitarianism vs Elitism
    # Culture Domain
    "I5": "ideology_ethical_pursuit",  # Progressivism vs Conservatism
    "I6": "ideology_church_state",  # Secularism vs Caesaropapism
    "I7": "ideology_cultural_value",  # Multiculturalism vs Nationalism
    # Diplomacy Domain
    "I8": "ideology_diplomatic_strategy",  # Internationalism vs Isolationism
    "I9": "ideology_military_force",  # Pacifism vs Militarism
    # Society Domain
    "I10": "ideology_social_development",  # Environmentalism vs Anthropocentrism
    "I11": "ideology_justice_orientation",  # Rehabilitationism vs Retributivism
    "I12": "ideology_personal_right",  # Libertarianism vs Authoritarianism
}

DOMAIN_RELEVANCE_COLS = {
    "R1": "relevance_politics",
    "R2": "relevance_economy",
    "R3": "relevance_culture",
    "R4": "relevance_diplomacy",
    "R5": "relevance_society",
}

FACET_RELEVANCE_COLS = {
    # Politics Domain Facets
    "R1-1-1": "relevance_political_regime",
    "R2-1-2": "relevance_state_structure",
    # Economy Domain Facets
    "R3-2-1": "relevance_economic_orientation",
    "R4-2-2": "relevance_economic_equality",
    # Culture Domain Facets
    "R5-3-1": "relevance_ethical_pursuit",
    "R6-3-2": "relevance_church_state",
    "R7-3-3": "relevance_cultural_value",
    # Diplomacy Domain Facets
    "R8-4-1": "relevance_diplomatic_strategy",
    "R9-4-2": "relevance_military_force",
    # Society Domain Facets
    "R10-5-1": "relevance_social_development",
    "R11-5-2": "relevance_justice_orientation",
    "R12-5-3": "relevance_personal_right",
}

ALL_RENAMES = {**IDEOLOGY_COLS, **DOMAIN_RELEVANCE_COLS, **FACET_RELEVANCE_COLS}

# Value mappings for meaningful labels
IDEOLOGY_VALUES = {0: "left", 1: "center", 2: "right"}
RELEVANCE_VALUES = {0: "unrelated", 1: "related"}

# Load MITweet dataset
df = pl.read_csv(_URL, null_values="-1")  # https://github.com/LST1836/MITweet
df = df.rename(ALL_RENAMES)

ideology_cols = list(IDEOLOGY_COLS.values())
relevance_cols = list(DOMAIN_RELEVANCE_COLS.values()) + list(
    FACET_RELEVANCE_COLS.values()
)

# Cast to integers for calculations
df = df.with_columns([pl.col(col).cast(pl.Int16) for col in ideology_cols])
df = df.with_columns([pl.col(col).cast(pl.Int16) for col in relevance_cols])

# Calculate partisan lean percentages before converting to strings
df = df.with_columns(
    num_left=pl.sum_horizontal(
        [(pl.col(col) == 0).cast(pl.Int8) for col in ideology_cols]
    ),
    num_center=pl.sum_horizontal(
        [(pl.col(col) == 1).cast(pl.Int8) for col in ideology_cols]
    ),
    num_right=pl.sum_horizontal(
        [(pl.col(col) == 2).cast(pl.Int8) for col in ideology_cols]
    ),
    num_non_null=pl.sum_horizontal(
        [(pl.col(col).is_not_null()).cast(pl.Int8) for col in ideology_cols]
    ),
)

# Pcts will be used to determine partisan lean
df = df.with_columns(
    pct_left=(pl.col("num_left") / pl.col("num_non_null")).round(2).fill_null(0.0),
    pct_center=(pl.col("num_center") / pl.col("num_non_null")).round(2).fill_null(0.0),
    pct_right=(pl.col("num_right") / pl.col("num_non_null")).round(2).fill_null(0.0),
)

# If 100% facets are left, right, center, or mixed, assign that label
df = df.with_columns(
    partisan_lean=pl.when(pl.col("pct_left") == 1.0)
    .then(pl.lit("LEFT"))
    .when(pl.col("pct_right") == 1.0)
    .then(pl.lit("RIGHT"))
    .when(pl.col("pct_center") == 1.0)
    .then(pl.lit("CENTER"))
    .when(
        pl.col("pct_left").is_null()
        & pl.col("pct_center").is_null()
        & pl.col("pct_right").is_null()
    )
    .then(pl.lit("NOT POLITICAL"))
    .otherwise(pl.lit("MIXED"))
)

# Convert numeric values to meaningful labels
df = df.with_columns(
    [pl.col(col).replace_strict(IDEOLOGY_VALUES, default=None) for col in ideology_cols]
)
df = df.with_columns(
    [
        pl.col(col).replace_strict(RELEVANCE_VALUES, default=None)
        for col in relevance_cols
    ]
)

# Select final columns for output
df = df.select(
    [
        pl.col("topic"),
        pl.col("tweet"),
        *[pl.col(col) for col in ideology_cols],
        *[pl.col(col) for col in relevance_cols],
        pl.col("partisan_lean"),
    ]
)

df_100 = df.sample(100, seed=195724)
df_100.write_csv("data/mitweet_sample_all_aspects.csv")
df_100 = df_100.select("topic", "tweet", "partisan_lean")
df_100.write_csv("data/mitweet_sample.csv")
