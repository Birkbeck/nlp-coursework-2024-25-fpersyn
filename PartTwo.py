from pathlib import Path
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np


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


def get_features(docs: list[str]) -> tuple[np.array, np.ndarray]:
    """
    Extract features from the corpus.

    Returns
        headers: A numpy array of feature headers.
        X: A numpy ndarray of extracted features.
    """
    vectoriser = TfidfVectorizer(stop_words="english", max_features=3000)
    features: np.ndarray = vectoriser.fit_transform(docs)
    headers: np.array = vectoriser.get_feature_names_out()

    logging.debug(f"features extracted (shape): {features.shape}")
    logging.debug(f"features extracted (excerpt): {headers[100:110]}")
    return headers, features


def inference_pipeline(model, X_test, y_test) -> np.array:
    """Generalised inference pipeline."""

    preds = model.predict(X_test)

    macro_f1 = f1_score(y_test, preds, average="macro")
    print("Macro-average F1 score", macro_f1)

    print("Classification report:")
    print(classification_report(y_test, preds))

    return preds


if __name__ == "__main__":
    logging.info("Started script part 2.")

    # question A - get dataset
    logging.info("Running code for part 2 question A.")
    df = get_dataset()
    df = filter_dataset(df)

    # question B - get features and train/test datasets
    logging.info("Running code for part 2 question B.")
    headers, X = get_features(df["speech"].to_list())
    y = df["party"].to_numpy()  # target
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=26)

    # question C - train/test models
    logging.info("Running code for part 2 question C.")

    logging.info("Running random forest model.")
    rf_clf = RandomForestClassifier(n_estimators=300)
    rf_clf.fit(X_train, y_train)
    inference_pipeline(rf_clf, X_test, y_test)

    logging.info("Running linear SVM model.")
    svm_clf =SVC(kernel='linear')
    svm_clf.fit(X_train, y_train)
    inference_pipeline(svm_clf, X_test, y_test)

    logging.info("Ended script part 2.")
