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
    with open("secrets/OPENAIKEY.txt.template", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read()
    client = OpenAI()
    return (client,)


@app.cell
def _(pl):
    # Load MITweet dataset
    df = pl.read_csv("data/mitweet_sample.csv")
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
    SIMPLE_PROMPT_TEMPLATE = """You are a political ideology classifier.

    Your job is to assign ONE label to the tweet below.

    LABELS (choose exactly one):

    LEFT
    - Supports progressive or liberal political positions
    - Criticizes conservatives or Republicans from a liberal perspective
    - Examples: BLM support, abortion rights, LGBTQ+ rights, climate action

    RIGHT
    - Supports conservative political positions
    - Criticizes liberals or Democrats from a conservative perspective
    - Examples: anti-wokeness, small government, nationalism, law-and-order rhetoric

    CENTER
    - Political but primarily informational, factual, or descriptive
    - Reports news, quotes statements, or summarizes events
    - No clear endorsement or attack on either side

    MIXED  (HIGH PRIORITY LABEL)
    You MUST choose MIXED if ANY of the following are true:
    - Both left-leaning and right-leaning groups are criticized
    - The tweet attacks “both sides”, “everyone”, or the political system broadly
    - The tweet supports one side but also criticizes it
    - Conflicting ideological signals are present

    NONE
    - Not political
    - Personal, cultural, entertainment, or everyday content


    DECISION RULES (FOLLOW IN ORDER):

    1) If the tweet is not political → NONE
    2) If political but purely factual or descriptive → CENTER
    3) If explicit praise or attack targets ONE side → LEFT or RIGHT
    4) If MORE THAN ONE ideological direction appears → MIXED

    CRITICAL OVERRIDE RULE (DO NOT IGNORE):

    If a tweet criticizes:
    - Democrats AND Republicans
    - liberals AND conservatives
    - “both sides”, “everyone”, “the left and the right”, or “the system”

    YOU MUST choose MIXED,
    even if one side is criticized more strongly.

    If there is ANY credible criticism of more than one ideological group,
    DO NOT choose LEFT or RIGHT.

    IMPORTANT:
    - MIXED overrides LEFT and RIGHT when ambiguity exists
    - Emotional tone does NOT determine ideology
    - Focus ONLY on who is being criticized or supported

    Tweet:
    \"\"\"{tweet}\"\"\"


    RESPONSE FORMAT:

    Return ONLY ONE word on its own line:

    LEFT
    CENTER
    RIGHT
    MIXED
    NONE
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
