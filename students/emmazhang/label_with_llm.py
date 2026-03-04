import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell
def _():
    import marimo as mo
    import re 
    import os
    import polars as pl
    from openai import OpenAI
    from tqdm import tqdm
    return OpenAI, mo, os, pl, re, tqdm


@app.cell
def _(OpenAI, os):
    # Set your API key
    # Make sure to create the file OPENAIKEY.txt before running this
    # (You can use the OPENAIKEY.txt.template file as a template)
    with open("secrets/OPENAIKEY.txt", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()
    client = OpenAI()
    return (client,)


@app.cell
def _(pl):
    # Load MITweet dataset
    df = pl.read_csv("/Users/emmazjy/Desktop/494/494-user-trajectories-ezfork/data/mitweet_sample.csv")
    return (df,)


@app.cell
def _(df):
    # Look at data
    df
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Labeling
    """)
    return


@app.cell
def _():
    SIMPLE_PROMPT_TEMPLATE = """You are labeling the partisan lean of a social media post.

    # LABELS (choose exactly ONE)
    - LEFT: supports progressive/left positions OR criticizes conservatives/Republicans/right-wing institutions.
    - RIGHT: supports conservative/right positions OR criticizes liberals/Democrats/left-wing institutions.
    - CENTER: mainly factual reporting, neutral informational updates, or balanced non-partisan wording with no clear endorsement/attack.
    - MIXED: contains BOTH left-leaning and right-leaning signals (e.g., endorses one side on one issue but criticizes it on another; or explicitly praises one side and also praises the other; or contains two distinct partisan frames).
    - NONE: not about politics or ideology (sports, entertainment, personal life, generic inspiration), or politics is only mentioned as a passing reference with no stance.

    # DECISION RULES
    1) First decide: is there ANY political content? If no -> NONE.
    2) If political: identify stance signals (endorsement/attack) and their target.
    3) Use MIXED ONLY when there are clear signals on BOTH sides. If it's merely ambiguous/unclear, do NOT use MIXED.
    4) If it's political but mostly neutral reporting -> CENTER.
    5) When uncertain between LEFT vs CENTER or RIGHT vs CENTER:
    - If there is explicit approval/criticism of a side -> LEFT/RIGHT
    - If it's mostly descriptive with no stance -> CENTER

    # TWEET
    {tweet}

    # OUTPUT (no extra text)
    <output>
    LEFT|CENTER|RIGHT|MIXED|NONE
    </output>
    """
    
    return (SIMPLE_PROMPT_TEMPLATE,)


@app.cell
def _(SIMPLE_PROMPT_TEMPLATE, client, df, pl, re, tqdm):
    def _parse_output(output_text: str) -> str:
        text = (output_text or "").strip()
        # Prefer the explicit <output> block if present
        m = re.search(r"<output>\s*(.*?)\s*</output>", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        return text.strip().upper()

    def _query_llm(row: dict) -> dict:
        tweet = row["tweet"]
        prompt = SIMPLE_PROMPT_TEMPLATE.format(tweet=tweet)
        resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
        output_text = getattr(resp, "output_text", "") or ""
        return output_text


    # Process rows with a for-loop
    results = []
    for row in tqdm(df.iter_rows(named=True), total=df.height):
        output_text = _query_llm(row)
        prediction = _parse_output(output_text)
        # Combine original row data with classification results
        result_row = {**row, **{"llm_output": output_text, "prediction": prediction}}
        results.append(result_row)

    # Convert results back to a DataFrame
    simple_predictions = pl.DataFrame(results)
    return (results,)


@app.cell
def _(mo):
    mo.md(r"""
    # Evaluation
    """)
    return


@app.cell
def _(pl, results):
    # Look at results
    pl.DataFrame(results)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Confusion Matrix
    """)
    return


@app.cell
def _(pl, results):
    crosstab = (
        pl.DataFrame(results)
        .group_by('partisan_lean', 'prediction')
        .len()
        .pivot(index="partisan_lean", on="prediction", values="len")
    )

    # Get prediction columns (everything except the index)
    prediction_columns = [col for col in crosstab.columns if col != "partisan_lean"]

    crosstab = (
        crosstab
        .with_columns(
            pl.concat_str([pl.lit("actually_"), pl.col("partisan_lean")]).alias("partisan_lean")
        )
        .rename({
            "partisan_lean": "actual_label",
            **{col: f"predicted_{col}" for col in prediction_columns}
        })
    )

    crosstab
    return


if __name__ == "__main__":
    app.run()
