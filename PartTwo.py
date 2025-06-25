from pathlib import Path

import pandas as pd


def get_dataset() -> pd.DataFrame:
    """Load the dataset."""

    csv_path = Path.cwd() / "p2-texts" / "hansard40000.csv"
    df = pd.read_csv(csv_path, header=0)

    print("dataframe size:", df.shape)
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
    print("TOP4 most common parties:", top4_most_common_parties)

    print("dataframe size after filtering:", df.shape)
    print(df.head())
    return df


if __name__ == "__main__":
    df = get_dataset()
    df = filter_dataset(df)
