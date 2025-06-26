from pathlib import Path
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import spacy


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
nlp = spacy.load("en_core_web_sm")


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


def get_features(docs: list[str], vectoriser = None) -> tuple[np.array, np.ndarray]:
    """
    Extract features from the corpus.

    Args:
        docs: A list of string documents.
        vectoriser: A scikit-learn vectoriser (optional).

    Returns
        headers: A numpy array of feature headers.
        X: A numpy ndarray of extracted features.
    """
    if not vectoriser:
        vectoriser = TfidfVectorizer(stop_words="english", max_features=3000)
    features: np.ndarray = vectoriser.fit_transform(docs)
    headers: np.array = vectoriser.get_feature_names_out()

    logging.debug(f"features extracted (shape): {features.shape}")
    logging.debug(f"features extracted (excerpt): {headers[100:110]}")
    return features


def inference_pipeline(model, X_test, y_test) -> np.array:
    """Generalised inference pipeline."""

    preds = model.predict(X_test)

    macro_f1 = f1_score(y_test, preds, average="macro")
    print("Macro-average F1 score", macro_f1)

    print("Classification report:")
    print(classification_report(y_test, preds))

    return preds


def custom_tokenizer(text: str) -> list[str]:
    """Custom tokenizer to tokenize political speeches."""

    # parse text
    tokens: spacy.tokens.Doc = nlp(text)
    logging.debug("Parsed document.")

    # ignore spaces and punctuation
    tokens = [t for t in tokens if not t.is_space and not t.is_punct]
    # ignore stop words
    tokens = [t for t in tokens if not t.is_stop]
    # only include open-class words (content-bearing)
    tokens = [t for t in tokens if t.pos_ in ["NOUN", "PROPN", "VERB", "ADJ", "ADV", "INTJ"]]
    # only use the base-form of each word (adjusts for capitalisation, tenses, singular/plural)
    tokens = [t.lemma_ for t in tokens]

    return tokens


if __name__ == "__main__":
    logging.info("Started script part 2.")

    # question A - get dataset
    logging.info("Running code for part 2 question A.")
    df = get_dataset()
    df = filter_dataset(df)

    # question B - get features and train/test datasets
    logging.info("Running code for part 2 question B.")
    y = df["party"].to_numpy()  # target
    X = get_features(df["speech"].to_list())
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=26)

    # question C - train/test models
    logging.info("Running code for part 2 question C.")
    rf_clf = RandomForestClassifier(n_estimators=300)
    svm_clf =SVC(kernel='linear')

    logging.info("Running random forest model (original feature set).")
    rf_clf.fit(X_train, y_train)
    inference_pipeline(rf_clf, X_test, y_test)

    logging.info("Running linear SVM model (original feature set).")
    svm_clf.fit(X_train, y_train)
    inference_pipeline(svm_clf, X_test, y_test)

    # question D - alt feature set using vectoriser with uni-/bi-/tri-grams enabled
    logging.info("Running code for part 2 question D.")
    alt_vectoriser = TfidfVectorizer(
        stop_words="english",
        max_features=3000,
        ngram_range=(1, 3)  # enables unigrams up to trigrams
    )
    X = get_features(df["speech"].to_list(), vectoriser=alt_vectoriser)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=26)

    logging.info("Running random forest model (alt feature set with bi-/trigrams).")
    rf_clf.fit(X_train, y_train)
    inference_pipeline(rf_clf, X_test, y_test)

    logging.info("Running linear SVM model (alt feature set with bi-/trigrams).")
    svm_clf.fit(X_train, y_train)
    inference_pipeline(svm_clf, X_test, y_test)

    # question E - alt feature set using custom tokeniser
    logging.info("Running code for part 2 question E.")
    custom_vectoriser = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 3),
        stop_words=None,  # handled in custom_tokenizer
        tokenizer=custom_tokenizer,
        token_pattern=None,  # to supress warning
        min_df=2  # focus on repeat word-use
    )
    X = get_features(df["speech"].to_list(), vectoriser=custom_vectoriser)
    X = SelectKBest(chi2, k=500).fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=26)

    logging.info("Running random forest model (alt feature set with custom tokeniser).")
    rf_clf.fit(X_train, y_train)
    inference_pipeline(rf_clf, X_test, y_test)

    logging.info("Running linear SVM model (alt feature set with custom tokeniser).")
    svm_clf.fit(X_train, y_train)
    inference_pipeline(svm_clf, X_test, y_test)

    logging.info("Ended script part 2.")
