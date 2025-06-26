from pathlib import Path
import logging

import pandas as pd


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def get_dataset() -> pd.DataFrame:
    """Load the dataset."""

    csv_path = Path.cwd() / "p2-texts" / "hansard40000.csv"
    df = pd.read_csv(csv_path, header=0)

    logging.debug(f"dataframe size: {df.shape}")
    print(df.head())
    return df


def filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the dataset."""

    # replace duplicate alt label
    df["party"] = df["party"].replace("Labour (Co-op)", "Labour")

    # remove speaker party value
    df = df[df["party"] != "Speaker"]

    # only keep top4 most common parties
    top4_most_common_parties = df['party'].value_counts()[:4].index.to_list()
    df = df[df["party"].isin(top4_most_common_parties)]
    logging.debug(f"TOP4 most common parties: {top4_most_common_parties}")

    # only keep speeches
    df = df[df["speech_class"] == "Speech"]

    # only keep long speeches
    df = df[df['speech'].str.len() >= 1000]

    logging.debug(f"dataframe size after filtering: {df.shape}")
    print(df.head())
    return df


if __name__ == "__main__":
    logging.info("Started script part 2.")

    # question A - get dataset
    logging.info("Running code for part 2 question A.")
    df = get_dataset()
    df = filter_dataset(df)

    logging.info("Ended script part 2.")
