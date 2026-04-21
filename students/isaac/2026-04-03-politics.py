import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns

    return Path, mo, pl


@app.cell
def _(Path):
    DATA_DIR = Path(".").absolute().parent.parent / "data"
    return (DATA_DIR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Should we use scores Mohsen sent?
    """)
    return


@app.cell
def _(DATA_DIR, pl):
    _renault_public = (
        pl.read_csv(DATA_DIR / "renault_partisanship_labels.csv", 
                    schema_overrides={"note_id": pl.String, "tweet_author_id": pl.String, "tweet_id": pl.String})
        .select("tweet_author_id", "party", "party_barbera", "party_mosleh", "party_gpt")
        .unique()
    )

    ######### Read data
    _user_ideology_barbera = (
        pl.concat([
            pl.read_csv(
                DATA_DIR / "mosleh/07-user_ideology-barbera.csv",
                schema_overrides={"id_str": pl.String}, null_values=["NA"],),
            pl.read_csv(
                DATA_DIR / "mosleh/07b-user_ideology-barbera-unhelpful.csv",
                schema_overrides={"id_str": pl.String}, null_values=["NA"],),])
        # There are some infinite and null values present, which are not valid
        .filter(pl.col("ideo").is_not_null() & ~pl.col("ideo").is_infinite())
        .drop("")
        .rename({"id_str": "tweet_author_id", "ideo": "partisan_score_barbera"})
    )
    _barbera_mosleh_gpt = (
        pl.read_excel(DATA_DIR / "mosleh/barbera_mosleh_gpt.xlsx")
        .drop("__UNNAMED__0")
        .rename({"partisan_gpt": "party_gpt_private", "partisan_perplexity": "party_perplexity_private"})
        .drop("barbera_accurate", "mosleh_accurate", "note_published_helpful")
    )
    _barbera_mosleh_perplexity = (
        pl.read_excel(DATA_DIR / "mosleh/barbera_mosleh_perplexity.xlsx")
        .drop("__UNNAMED__0")
        .rename({"partisan_perplexity": "party_perplexity_private"})
        .drop("barbera_accurate", "mosleh_accurate", "note_published_helpful")
    )


    ######## Concat scores from different data sets, then categorize based on scores
    _barbera_mosleh = (
        pl.concat([
            _user_ideology_barbera    .select(["tweet_author_id", "partisan_score_barbera"])
                .with_columns(partisan_score=pl.lit(None).cast(pl.Float64)),
            _barbera_mosleh_gpt       .select(["tweet_author_id", "partisan_score_barbera", "partisan_score"]),
            _barbera_mosleh_perplexity.select(["tweet_author_id", "partisan_score_barbera", "partisan_score"]),
        ])
        .with_columns(
            party_barbera_private = 
                pl.when(pl.col("partisan_score_barbera") > 1).then(pl.lit("republican"))
                    .when(pl.col("partisan_score_barbera").is_not_null()).then(pl.lit("democrat")),
            party_mosleh_private = 
                    pl.when(pl.col("partisan_score") > 0).then(pl.lit("republican"))
                        .when(pl.col("partisan_score").is_not_null()).then(pl.lit("democrat")))
        .drop("partisan_score_barbera", "partisan_score")
        # Get first non null value, which will be only unique non-null value
        .group_by("tweet_author_id").agg(
            party_barbera_private = pl.col("party_barbera_private").filter(pl.col("party_barbera_private").is_not_null()).first(),
            party_mosleh_private = pl.col("party_mosleh_private").filter(pl.col("party_mosleh_private").is_not_null()).first(),
            nunique_barbera = pl.col("party_barbera_private").filter(pl.col("party_barbera_private").is_not_null()).n_unique(),
            nunique_mosleh = pl.col("party_mosleh_private").filter(pl.col("party_mosleh_private").is_not_null()).n_unique(),
        )
    )

    # Make sure there aren't any contradictory labels for the same author
    assert _barbera_mosleh.filter(pl.col("nunique_barbera") > 1).is_empty(), "There are contradictory Barbera labels for the same author"
    assert _barbera_mosleh.filter(pl.col("nunique_mosleh") > 1).is_empty(), "There are contradictory Mosleh labels for the same author"

    ######### Concat AI labels from different datasets
    _ai = (
        pl.concat([
            _barbera_mosleh_perplexity.select(["tweet_author_id", "party_perplexity_private"])
                .with_columns(party_gpt_private=pl.lit(None).cast(pl.String)),
            _barbera_mosleh_gpt       .select(["tweet_author_id", "party_perplexity_private", "party_gpt_private"]),
        ])
        .group_by("tweet_author_id").agg(
            party_perplexity_private = pl.col("party_perplexity_private").filter(pl.col("party_perplexity_private").is_not_null()).first(),
            party_gpt_private = pl.col("party_gpt_private").filter(pl.col("party_gpt_private").is_not_null()).first(),
            nunique_perplexity = pl.col("party_perplexity_private").filter(pl.col("party_perplexity_private").is_not_null()).n_unique(),
            nunique_gpt = pl.col("party_gpt_private").filter(pl.col("party_gpt_private").is_not_null()).n_unique(),
        )
    )

    # Make sure there aren't any contradictory labels for the same author
    assert _ai.filter(pl.col("nunique_perplexity") > 1).is_empty(), "There are contradictory Perplexity labels for the same author"
    assert _ai.filter(pl.col("nunique_gpt") > 1).is_empty(), "There are contradictory GPT labels for the same author"



    # From SI (they use 218k num in main paper): "The sample includes 169,270 Community Notes when using only [1], 229,393 when using only [2], and 218,382 when combining [1], [2], and the LLM (assigning the majority label as the final classification when at least two of the three methods agree)."

    ####### Join everything together
    author_ideologies = (
        _barbera_mosleh.select("tweet_author_id", "party_mosleh_private", "party_barbera_private")
        .join(_ai.select("tweet_author_id", "party_gpt_private", "party_perplexity_private"), 
              on=["tweet_author_id"], how="full", validate="1:1", coalesce=True)
        .join(_renault_public, on=["tweet_author_id"], how="full", validate="1:1", coalesce=True)
        .with_columns(party_private=
                      pl.when(pl.col("party_barbera_private") == pl.col("party_mosleh")).then(pl.col("party_barbera_private"))
                      .when(pl.col("party_barbera_private") == pl.col("party_gpt_private")).then(pl.col("party_barbera_private"))
                      .when(pl.col("party_gpt_private") == pl.col("party_mosleh_private")).then(pl.col("party_gpt_private"))
                      .otherwise(pl.lit("unknown")))
    )
    return (author_ideologies,)


@app.cell
def _(author_ideologies, pl):
    private_public_diff = (
        author_ideologies
        .filter(pl.col("party_private") != pl.col("party"))
        .filter(pl.col("party_private") != pl.lit("unknown"))
    )

    private_public_diff
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are only 5 cases where the private classification differs from the public one, and the private one is not "unknown". We therefore should prefer the public one in all cases.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Make partisanship data
    """)
    return


@app.cell
def _(DATA_DIR, pl):
    def _enrich_with_scores(
        notes: pl.DataFrame, ratings: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        scores = pl.read_parquet(DATA_DIR / "2026-02-03-scored_notes.parquet")
        scores = scores.with_columns(noteId=pl.col("noteId").cast(pl.String))
        I_AND_F_COLUMNS = {
            "CoreModel (v1.1)": ("coreNoteIntercept", "coreNoteFactor1"),
            "ExpansionModel (v1.1)": ("expansionNoteIntercept", "expansionNoteFactor1"),
            "ExpansionPlusModel (v1.1)": ("expansionPlusNoteIntercept", "expansionPlusNoteFactor1"),
            "GroupModel01 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel02 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel03 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel04 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel05 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel06 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel07 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel08 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel09 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel10 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel11 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel12 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel13 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "GroupModel14 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
            "MultiGroupModel01 (v1.0)": ("multiGroupNoteIntercept", "multiGroupNoteFactor1"),
            "TopicModel01 (v1.0)": ("topicNoteIntercept", "topicNoteFactor1"),
            "TopicModel02 (v1.0)": ("topicNoteIntercept", "topicNoteFactor1"),
            "TopicModel03 (v1.0)": ("topicNoteIntercept", "topicNoteFactor1"),
            "ScoringDriftGuard (v1.0)": (None, None),
            "NmrDueToMinStableCrhTime (v1.0)": (None, None),
            "InsufficientExplanation (v1.0)": (None, None),
        }

        scores = (
            scores
            .with_columns(scoreCreatedAtDt=pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms"))
            # Infer the pre drift guard model
            .with_columns(
                preDriftModel=pl.when(pl.col("decidedBy").str.contains("ScoringDriftGuard"))
                                .then(pl.col("metaScorerActiveRules").str.split(",").list[-2])
                                .otherwise(pl.col("decidedBy"))
            )
            # Retrieve the intercept from the inferred model
            .with_columns(
                noteFinalIntercept=pl.coalesce([
                    pl.when(pl.col("preDriftModel").str.starts_with(prefix))
                      .then(pl.col(intercept_col))
                    for prefix, (intercept_col, _) in I_AND_F_COLUMNS.items()
                    if intercept_col is not None
                ]),
                # Retrieve the factor from the inferred model
                noteFinalFactor=pl.coalesce([
                    pl.when(pl.col("preDriftModel").str.starts_with(prefix))
                      .then(pl.col(factor_col))
                    for prefix, (_, factor_col) in I_AND_F_COLUMNS.items()
                    if factor_col is not None
                ]),
            )
            .rename({"finalRatingStatus": "noteFinalRatingStatus"})
            .select("noteId", "noteFinalRatingStatus", "numRatings", "decidedBy", "noteFinalIntercept", "noteFinalFactor")
        )

        notes = notes.join(scores, on="noteId", how="left", coalesce=True, validate="1:1")
        return notes


    def _enrich_with_post_data(
        _notes: pl.DataFrame,
    ) -> pl.DataFrame:
        _our_post_data = (
            pl.scan_parquet("/data/cn_archive/derivatives/20260227_raw_posts.parquet")
            .select("post_id", "author_id", "lang")
            .filter(pl.col("author_id").is_not_null() | pl.col("lang").is_not_null())
            .collect()
            # Select first non null lang/author_id
            .group_by("post_id")
            .agg(
                tweet_author_id = pl.col("author_id").filter(pl.col("author_id").is_not_null()).first(),
                tweet_lang = pl.col("lang").filter(pl.col("lang").is_not_null()).first(),
                n_non_null_author_id = pl.col("author_id").filter(pl.col("author_id").is_not_null()).n_unique(),
                n_non_null_lang = pl.col("lang").filter(pl.col("lang").is_not_null()).n_unique()))

        _renault_post_data = (
            pl.read_csv(
                DATA_DIR / "renault_partisanship_labels.csv", 
                schema_overrides={"tweet_author_id": pl.String, "tweet_id": pl.String})
            .group_by("tweet_id")
            .agg(
                tweet_author_id = pl.col("tweet_author_id").filter(pl.col("tweet_author_id").is_not_null()).first(),
                n_non_null_author_id = pl.col("tweet_author_id").filter(pl.col("tweet_author_id").is_not_null()).n_unique())
            .with_columns(tweet_lang=pl.lit("en")) # Renault data only contains English tweets (see M&M)
        )

        # Make sure no contradictory values within a dataset
        assert _our_post_data.filter(pl.col("n_non_null_author_id") > 1).is_empty()
        assert _our_post_data.filter(pl.col("n_non_null_lang") > 1).is_empty()
        assert _renault_post_data.filter(pl.col("n_non_null_author_id") > 1).is_empty()

        # Join renault/our data
        _post_data = (
            _our_post_data.select("post_id", "tweet_author_id", "tweet_lang")
            .join(_renault_post_data.select("tweet_id", "tweet_author_id", "tweet_lang"), 
                left_on="post_id", right_on="tweet_id",
                how="full", validate="1:1", suffix="_renault")
        )

        # Make sure no contradictory values between datasets
        assert _post_data.filter(
            pl.col("tweet_author_id").is_not_null() & pl.col("tweet_author_id_renault").is_not_null() 
            & (pl.col("tweet_author_id") != pl.col("tweet_author_id_renault"))).is_empty()
        assert _post_data.filter(
            pl.col("tweet_lang").is_not_null() & pl.col("tweet_lang_renault").is_not_null() 
            & (pl.col("tweet_lang") != pl.col("tweet_lang_renault"))).is_empty()

        # Coalesce 
        _post_data = (
            _post_data.with_columns(
                post_id = pl.coalesce(["post_id", "tweet_id"]),
                tweet_author_id = pl.coalesce(["tweet_author_id", "tweet_author_id_renault"]),
                tweet_lang = pl.coalesce(["tweet_lang", "tweet_lang_renault"]))
            .rename({"post_id": "tweetId"})
            .select("tweetId", "tweet_author_id", "tweet_lang")
        )
    
        # Join notes to post data
        _notes = (
            _notes
            .join(_post_data, on="tweetId", how="left", coalesce=True, validate="m:1")
        )
        return _notes


    def _enrich_with_renault_author_party(
        notes: pl.DataFrame,
    ) -> pl.DataFrame:
        _renault = pl.read_csv(
            DATA_DIR / "renault_partisanship_labels.csv", 
            schema_overrides={"note_id": pl.String, "tweet_author_id": pl.String, "tweet_id": pl.String})

        _renault_authors = _renault.select(["tweet_author_id", "party"]).unique().rename({"party": "author_party"})
        _renault_posts = _renault.select(["tweet_id", "party"]).unique().rename({"party": "post_party"})
        _renault_notes = _renault.select(["note_id", "party"]).unique().rename({"party": "note_party"})

        notes = (
            notes
            .join(
                _renault_authors, on="tweet_author_id", 
                how="left", coalesce=True, validate="m:1")
            .join(
                _renault_posts, left_on="tweetId", right_on="tweet_id", 
                how="left", coalesce=True, validate="m:1")
            .join(
                _renault_notes, left_on="noteId", right_on="note_id", 
                how="left", coalesce=True, validate="m:1")
        )

        # Make sure there aren't contradictory labels for the same note/author/post
        assert notes.filter(
            pl.col("author_party").is_not_null() & pl.col("post_party").is_not_null() 
            & (pl.col("author_party") != pl.col("post_party"))).is_empty()
        assert notes.filter(
            pl.col("author_party").is_not_null() & pl.col("note_party").is_not_null() 
            & (pl.col("author_party") != pl.col("note_party"))).is_empty()
        assert notes.filter(
            pl.col("post_party").is_not_null() & pl.col("note_party").is_not_null() 
            & (pl.col("post_party") != pl.col("note_party"))).is_empty()

        # Coalesce
        notes = notes.with_columns(
            tweet_author_party=pl.coalesce([pl.col("author_party"), pl.col("post_party"), pl.col("note_party")])
        ).drop("author_party", "post_party", "note_party")
        return notes

    _raw_notes=pl.read_parquet(DATA_DIR / "2026-02-03/notes.parquet")
    _raw_ratings=pl.read_parquet(DATA_DIR / "2026-02-03/noteRatings.parquet")

    notes = _raw_notes.with_columns(
        tweetId=pl.col("tweetId").cast(pl.String),
        noteId=pl.col("noteId").cast(pl.String),
    )
    ratings = _raw_ratings.with_columns(
        noteId=pl.col("noteId").cast(pl.String),
    )
    notes =_enrich_with_scores(notes=notes, ratings=ratings)
    notes = _enrich_with_post_data(_notes=notes)
    notes = _enrich_with_renault_author_party(notes=notes)
    return


if __name__ == "__main__":
    app.run()
