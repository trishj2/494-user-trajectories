#!/usr/bin/env python3
"""Strip extra columns from filtered Community Notes data to match scoring schema.

This script reads the filtered TSV files and removes any columns not expected
by the scoring algorithm, ensuring compatibility without modifying the scoring code.

Usage:
    python strip_extra_columns.py
"""

from pathlib import Path
import polars as pl

# Expected columns for each data type (matching scoring/src/scoring/constants.py)
EXPECTED_COLUMNS = {
    "notes": [
        "noteId",
        "noteAuthorParticipantId",
        "createdAtMillis",
        "tweetId",
        "classification",
        "believable",
        "harmful",
        "validationDifficulty",
        "misleadingOther",
        "misleadingFactualError",
        "misleadingManipulatedMedia",
        "misleadingOutdatedInformation",
        "misleadingMissingImportantContext",
        "misleadingUnverifiedClaimAsFact",
        "misleadingSatire",
        "notMisleadingOther",
        "notMisleadingFactuallyCorrect",
        "notMisleadingOutdatedButNotWhenWritten",
        "notMisleadingClearlySatire",
        "notMisleadingPersonalOpinion",
        "trustworthySources",
        "summary",
        "isMediaNote",
    ],
    "ratings": [
        "noteId",
        "raterParticipantId",
        "createdAtMillis",
        "version",
        "agree",
        "disagree",
        "helpful",
        "notHelpful",
        "helpfulnessLevel",
        "helpfulOther",
        "helpfulInformative",
        "helpfulClear",
        "helpfulEmpathetic",
        "helpfulGoodSources",
        "helpfulUniqueContext",
        "helpfulAddressesClaim",
        "helpfulImportantContext",
        "helpfulUnbiasedLanguage",
        "notHelpfulOther",
        "notHelpfulIncorrect",
        "notHelpfulSourcesMissingOrUnreliable",
        "notHelpfulOpinionSpeculationOrBias",
        "notHelpfulMissingKeyPoints",
        "notHelpfulOutdated",
        "notHelpfulHardToUnderstand",
        "notHelpfulArgumentativeOrBiased",
        "notHelpfulOffTopic",
        "notHelpfulSpamHarassmentOrAbuse",
        "notHelpfulIrrelevantSources",
        "notHelpfulOpinionSpeculation",
        "notHelpfulNoteNotNeeded",
        "ratedOnTweetId",
    ],
    "notes-status-history": [
        "noteId",
        "noteAuthorParticipantId",
        "createdAtMillis",
        "timestampMillisOfFirstNonNMRStatus",
        "firstNonNMRStatus",
        "timestampMillisOfCurrentStatus",
        "currentStatus",
        "timestampMillisOfLatestNonNMRStatus",
        "mostRecentNonNMRStatus",
        "timestampMillisOfStatusLock",
        "lockedStatus",
        "timestampMillisOfRetroLock",
        "currentCoreStatus",
        "currentExpansionStatus",
        "currentGroupStatus",
        "currentDecidedBy",
        "currentModelingGroup",
    ],
    "user-enrollment": [
        "participantId",
        "enrollmentState",
        "successfulRatingNeededToEarnIn",
        "timestampOfLastStateChange",
        "timestampOfLastEarnOut",
        "modelingPopulation",
        "modelingGroup",
    ],
}

# Get script directory and construct path to filtered data
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
FILTERED_DATA = PROJECT_ROOT.parent.parent / "data" / "filtered" / "2023-10"


def strip_columns(input_dir: Path, expected_cols: list[str]) -> None:
    """Strip extra columns from all TSV files in a directory."""
    if not input_dir.exists():
        print(f"  Directory not found: {input_dir}")
        return

    for tsv in sorted(input_dir.glob("*.tsv")):
        df = pl.read_csv(tsv, separator="\t", infer_schema_length=0)
        current_cols = df.columns

        # Find columns to keep (preserve order from expected_cols)
        cols_to_keep = [c for c in expected_cols if c in current_cols]
        extra_cols = [c for c in current_cols if c not in expected_cols]

        if extra_cols:
            print(f"  {tsv.name}: removing {extra_cols}")
            df = df.select(cols_to_keep)
            df.write_csv(tsv, separator="\t")
        else:
            print(f"  {tsv.name}: no extra columns")


def main():
    print("Stripping extra columns from filtered data...\n")

    for data_type, expected_cols in EXPECTED_COLUMNS.items():
        input_dir = FILTERED_DATA / data_type
        print(f"Processing {data_type}/")
        strip_columns(input_dir, expected_cols)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
