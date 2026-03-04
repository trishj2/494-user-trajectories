import marimo

__generated_with = "0.18.1"
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
    SIMPLE_PROMPT_TEMPLATE = """# TWEET
    {tweet}

    # ANALYSIS INSTRUCTIONS

    Use chain-of-thought reasoning to classify this tweet's partisan lean.

    **Step 1: Summarize the tweet's argument**
    What is this tweet claiming, advocating, or criticizing?

    **Step 2: Summarize context**
    What relevant background information is necessary to understand this tweet's ideological positioning?

    **Step 3: Determine direction**
    Based on the tweet and context, which partisan lean does the tweet align with?

    # RESPONSE FORMAT

    <analysis>
    **Tweet's main argument:** [1-2 sentences]

    **Context:** [1-2 sentence]

    **Directional assessment:** [Direction] because [1-2 sentence reason]
    </analysis>

    <output>
    [LEFT/CENTER/RIGHT/MIXED]
    </output>

    Or if tweet is not political:
    <output>
    NONE
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
