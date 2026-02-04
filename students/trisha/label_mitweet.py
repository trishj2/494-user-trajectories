import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    return


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
    with open("/Users/trishajanappareddi/Desktop/494-user-trajectories/students/trisha/secret/OPENAIKEY.txt", "r") as f:
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
    SIMPLE_PROMPT_TEMPLATE = """You are labeling the political ideology expressed or clearly implied by the AUTHOR of a tweet.

    # TWEET
    {tweet}

    # TASK
    Classify the tweet's partisan lean using exactly ONE label:
    LEFT, CENTER, RIGHT, NONE

    # LABEL GUIDELINES
    - LEFT: supportive of progressive/liberal positions (e.g., stronger regulation, pro-choice, expanded social programs, critiques of conservatives/Republicans from the left).
    - RIGHT: supportive of conservative positions (e.g., lower taxes, stricter immigration, pro-life, gun rights, critiques of liberals/Democrats from the right).
    - CENTER: explicitly moderate/centrist stance (e.g., bipartisan framing, rejects extremes, balanced tradeoffs) with a clear stance.
    - NONE: not political OR no clear stance in the text.

    # RULES (IMPORTANT)
    - Label the AUTHOR’S stance, not the topic.
    - Do not infer ideology from tone, profanity, or identity words alone.
    - If the tweet is mostly a headline, quote, or vague reaction with no stance -> NONE.
    - Output must be exactly one label.

    # RESPONSE FORMAT
    <output>
    LABEL
    </output>
    """
    return (SIMPLE_PROMPT_TEMPLATE,)


@app.cell
def _(SIMPLE_PROMPT_TEMPLATE, client, df, re, tqdm):
    def _parse_output(output_text: str) -> str:
        text = (output_text or "").strip()
        m = re.search(r"<output>\s*(.*?)\s*</output>", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        return text 


    def _query_llm(row: dict) -> str:
        tweet = row["tweet"]
        prompt = SIMPLE_PROMPT_TEMPLATE.format(tweet=tweet)

        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )

        output_text = getattr(resp, "output_text", "") or ""
        return output_text 

    results = []
    for row in tqdm(df.iter_rows(named=True), total=df.height):
        llm_out = _query_llm(row)
        pred = _parse_output(llm_out)

        results.append({
            **row,
            "llm_output": llm_out,
            "prediction": pred
        })

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
    simple_predictions = pl.DataFrame(results)

    # Look at results
    simple_predictions
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
