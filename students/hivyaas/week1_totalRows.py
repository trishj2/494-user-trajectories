import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import os
    return os, pd


@app.cell
def _(pd):
    df = pd.read_csv("data/cn_data/notes-00000.tsv", sep='\t')
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    print("Total number of rows:" , len(df.index))
    return


@app.cell
def _(df):
    filtered_df = df[df["createdAtMillis"] <= 1698817568000] #used an online epoch convertor to get the epoch of Nov 1st, 2023
    filtered_df.head()
    return (filtered_df,)


@app.cell
def _(filtered_df):
    len(filtered_df)
    return


@app.cell
def _(filtered_df):
    filtered_df.reset_index(drop='true')
    return


@app.cell
def _(filtered_df):
    fdf = filtered_df.drop(columns=['isCollaborativeNote'])
    return (fdf,)


@app.cell
def _(fdf):
    fdf.head()
    return


@app.cell
def _(fdf):
    fdf.to_csv("/Users/himani/devsrc/494-user-trajectories/data/filtered-data/notes-00000.tsv", index=False, sep='\t')
    return


@app.cell
def _(fdf):
    fdf.head()
    return


@app.cell
def _(pd):
    rt1 = pd.read_csv("data/cn_data/rawRatings/ratings-00007.tsv", sep='\t') #do we need to download all of the ratings files. i only did 1 of each kind of file
    return (rt1,)


@app.cell
def _(rt1):
    rt1.dtypes
    return


@app.cell
def _(rt1):
    rt2 = rt1[rt1["createdAtMillis"] <= 1698817568000]
    rt2.dtypes
    return (rt2,)


@app.cell
def _(rt2):
    rt3 = rt2.drop(columns=['ratingSourceBucketed'])
    rt3.dtypes
    return (rt3,)


@app.cell
def _(pd, rt1, rt3):
    rts = [rt1, rt3]
    final_df = pd.concat(rts, ignore_index=True)
    final_df.to_csv("/Users/himani/devsrc/494-user-trajectories/data/filtered-data/test.tsv", index=False, sep='\t', mode='a')
    return


@app.cell
def _(pd):
    test = pd.read_csv("data/filtered-data/test.tsv", sep='\t', low_memory=False)
    return (test,)


@app.cell
def _(test):
    test.dtypes
    return


@app.cell
def _(pd):
    def filterRatings(ogFile):
        print(ogFile)
        df = pd.read_csv(ogFile, sep='\t')
        ratings2023 = df[df["createdAtMillis"] <= 1698817568000]
        ratings2023Drop = ratings2023.drop(columns=['ratingSourceBucketed'])
        return ratings2023Drop
    return


@app.cell
def _(os, pd):
    directory = os.fsencode("data/cn_data/rawRatings")

    allRatings = []
    for file in os.listdir(directory):
        ogFile = os.fsdecode(file)
        print(ogFile)
        df = pd.read_csv("data/cn_data/rawRatings/" + ogFile, sep='\t')
        ratings2023 = df[df["createdAtMillis"] <= 1698817568000]
        ratings2023Drop = ratings2023.drop(columns=['ratingSourceBucketed'])
        #r = filterRatings("data/cn_data/rawRatings/" + filename)
        allRatings.append(ratings2023Drop)

    finalRatings = pd.concat(allRatings, ignore_index=True)
    finalRatings.to_csv("/Users/himani/devsrc/494-user-trajectories/data/filtered-data/ratings-00000.tsv", index=False, sep='\t')
    return df, finalRatings


@app.cell
def _(finalRatings):
    finalRatings.to_csv("/Users/himani/devsrc/494-user-trajectories/data/filtered-data/ratings-00000.tsv", index=False, sep='\t')
    return


@app.cell
def _(finalRatings):
    finalRatings.head()
    return


@app.cell
def _(pd):
    allFilteredRatings = pd.read_csv("data/filtered-data/ratings-00000.tsv", sep='\t', low_memory=False)
    allFilteredRatings.head()
    return (allFilteredRatings,)


app._unparsable_cell(
    r"""
     filterRatings(\"data/cn_data/ratings-00007.tsv\", \"ratings-00007.tsv\")
    """,
    name="_"
)


@app.cell
def _(allFilteredRatings):
    allFilteredRatings.dtypes
    return


@app.cell
def _(rt1):
    rt1.head()
    len(rt1)
    return


@app.cell
def _(rt1):
    frt1 = rt1[rt1["createdAtMillis"] <= 1698817568000]
    len(frt1)
    return (frt1,)


@app.cell
def _(frt1):
    frt = frt1.drop(columns=['ratingSourceBucketed'])
    return (frt,)


@app.cell
def _(frt):
    len(frt)
    return


@app.cell
def _(frt):
    frt.to_csv("/Users/himani/devsrc/494-user-trajectories/data/filtered-data/ratings-00000.tsv", index=False, sep='\t')
    return


@app.cell
def _(pd):
    nsh = pd.read_csv("data/cn_data/noteStatusHistory-00000.tsv", sep='\t')
    return (nsh,)


@app.cell
def _(nsh):
    len(nsh)
    return


@app.cell
def _(nsh):
    nshf = nsh[nsh['createdAtMillis'] < 1698817568000]
    len(nshf)
    return (nshf,)


@app.cell
def _(nshf):
    dnsh = nshf.drop(columns=['timestampMillisOfMostRecentStatusChange', 'timestampMillisOfNmrDueToMinStableCrhTime', 'currentMultiGroupStatus','currentModelingMultiGroup', 'timestampMinuteOfFinalScoringOutput', 'timestampMillisOfFirstNmrDueToMinStableCrhTime'])
    return (dnsh,)


@app.cell
def _(dnsh):
    dnsh.head()
    return


@app.cell
def _(dnsh):
    dnsh.to_csv("/Users/himani/devsrc/494-user-trajectories/data/filtered-data/noteStatusHistory-00000.tsv", index=False, sep='\t')
    return


@app.cell
def _(pd):
    user = pd.read_csv("data/cn_data/userEnrollment-00000.tsv", sep='\t')
    return (user,)


@app.cell
def _(user):
    len(user)
    user.head()
    return


@app.cell
def _(user):
    fuser = user.drop(columns=['numberOfTimesEarnedOut']) #do we need to filter these
    return (fuser,)


@app.cell
def _(fuser):
    fuser.head()
    return


@app.cell
def _(fuser):
    fuser.to_csv("/Users/himani/devsrc/494-user-trajectories/data/filtered-data/userEnrollment-00000.tsv", index=False, sep='\t')
    return


@app.cell
def _(pd):
    bat = pd.read_csv("data/cn_data/batSignals-00000.tsv", sep='\t')
    return (bat,)


@app.cell
def _(bat):

    len(bat)
    return


@app.cell
def _(bat):
    bat.head()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
