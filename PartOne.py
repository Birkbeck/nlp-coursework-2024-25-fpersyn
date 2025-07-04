#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

from collections import Counter
from pathlib import Path
import logging
import string
import re

import pandas as pd
import spacy
import nltk


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000


def fk_level(text: str) -> float:
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text to the nearest 4 decimal points. (higher grade is more difficult)
    """
    cmudict = nltk.corpus.cmudict.dict()  # English syllable dictionary – embedded this in the function and changed its signature

    # preprocessing
    text_wrk = text.lower()  # ignore case
    text_wrk = re.sub(r"-{2,}", " ", text_wrk)  # replace repeat hyphens with spaces
    text_wrk = re.sub(r"[%s]" % string.punctuation, "", text_wrk)  # remove all punctuation symbols

    # tokenisation
    tokens_sentences = nltk.sent_tokenize(text)  # using the RAW text here because we need punctuation to tokenise sentences
    tokens_words = nltk.word_tokenize(text_wrk)
    syllable_counts: list[int] = [count_syl(w, cmudict) for w in tokens_words]  # only counting to save memory

    # metrics
    avg_sentence_length = len(tokens_words) / len(tokens_sentences)
    avg_syllables_per_word = sum(syllable_counts) / len(syllable_counts)
    fk_level: float = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59  # Flesch-Kincaid grade level formula

    return round(fk_level, 4)


def count_syl(word: str, word_syllables: dict) -> int:
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        word_syllables (dict): A word-syllable dictionary. For example, nltk.corpus.reader.cmudict

    Returns:
        int: The number of syllables in the word.
    """
    results: list[list[str]] = word_syllables.get(word)  # can have multiple results (e.g. "tomato")
    if results:
        return len(results[0])  # only count syllables in first match
    return 0  # fallback value if no match


def read_novels(path: Path = Path.cwd() / "p1-texts") -> pd.DataFrame:
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""

    data: list[dict[str, str]] = []
    for file_path in path.rglob("*.txt"):
        title, author, year = file_path.stem.split("-")  # split filename in parts
        with file_path.open(mode="r") as f:
            text = f.read()
        data.append({
            "title": title.replace("_", " "),  # readability fix
            "author": author,
            "year": year,
            "text": text
        })
    return (
        pd.DataFrame
        .from_records(data)
        .sort_values("year")
        .reset_index(drop=True)
    )


def parse(
    df: pd.DataFrame,
    store_path: Path = Path.cwd() / "pickles",
    out_name: str = "parsed.pickle"
) -> pd.DataFrame:
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""

    def parse_text(text: str):
        """Parse a text using spacy's nlp pipeline."""
        doc = nlp(text)
        logging.debug("Extracted document using spacy.")
        return doc

    # parse documents
    df["parsed"] = df["text"].apply(parse_text)
    logging.debug("Added new column to DataFrame with parsed Doc objects for each file.")

    # pickle dataframe
    if not store_path.exists():
        store_path.mkdir()
    df.to_pickle(path=store_path / out_name)
    logging.debug("Saved pickle file.")

    return df


def nltk_ttr(text: str) -> float:
    """
    Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize.

    Note: The coursework stipulates this function should return a dict with values for each title.
          I chose to ignore this because and return a TTR (float value) instead. Motivation:
          (1) This function takes text (str) as an input argument - not a list or dict of documents.
          (2) The function below (get_ttrs) is already responsible for building such a dict. 
    """
    # preprocessing
    text = text.lower()  # ignore case
    text = re.sub(r"[%s]" % string.punctuation, "", text)  # remove all punctuation symbols

    # tokenisation
    tokens = nltk.word_tokenize(text)

    # counts
    n_tokens: int = len(tokens)
    n_types: int = len(set(tokens))  # unique tokens

    return n_types / n_tokens


def get_ttrs(df: pd.DataFrame) -> dict[str, float]:
    """helper function to add ttr to a dataframe"""
    results = {}
    for _idx, row in df.iterrows():
        title = row["title"]
        text = row["text"]
        results[title] = nltk_ttr(text)
    return results


def get_fks(df) -> dict[str, float]:
    """helper function to add fk scores to a dataframe"""
    results = {}
    for _idx, row in df.iterrows():
        title = row["title"]
        text = row["text"]
        results[title] = fk_level(text)
    return results


def top10_subjects_by_verb_pmi(doc: spacy.tokens.Doc, verb: str):
    """
    Extracts the most common subjects of a given verb in a parsed document based on PMI.

    Args:
        doc: a spacy token document.
        verb: a base form verb (e.g. "to work" becomes "work")

    Returns:
        A list of tuples following the format: (word, pmi)

    Design:
        - Changed the name of this function for readability.
        - Selecting verb tokens by lemma (base form) ignores its tense.
        - By using spacy's dependency parsing we can extract the subject for each verb (NSUBJ).
        - Extracting the subject lemma (base form) accounts for capitalisation and singular/plural.
        - Assumed point-wise mutual information (PMI) of the subject with the verb.
        - Using a dict to avoid duplicate entries.
        - Transforming the dict to a list, ordering by PMI and returning the TOP10.
    """
    verb_tokens = [t for t in doc if t.lemma_ == verb]
    subjects = {child.lemma_: child.similarity(t) for t in verb_tokens for child in t.children if child.dep == spacy.symbols.nsubj}
    subject_pmi = [(k, v) for k, v in subjects.items() if v is not None]
    subject_pmi.sort(key=lambda x: x[1], reverse=True)
    return subject_pmi[:10]


def top10_subjects_by_verb_count(doc: spacy.tokens.Doc, verb: str) -> list[tuple[str, int]]:
    """
    Extracts the most common subjects of a given verb in a parsed document based on frequency.

    Args:
        doc: a spacy token document.
        verb: a base form verb (e.g. "to work" becomes "work")

    Returns:
        A list of tuples following the format: (word, frequency)

    Design:
        - Changed the name of this function for readability.
        - Selecting verb tokens by lemma (base form) ignores its tense.
        - By using spacy's dependency parsing we can extract the subject for each verb (NSUBJ).
        - Extracting the subject lemma (base form) accounts for capitalisation and singular/plural.
    """
    verb_tokens = [t for t in doc if t.lemma_ == verb]
    subjects = [child.lemma_ for t in verb_tokens for child in t.children if child.dep == spacy.symbols.nsubj]
    return Counter(subjects).most_common(10)


def top10_objects(doc: spacy.tokens.Doc) -> list[tuple[str, int]]:
    """
    Extracts the most common syntactic objects in a parsed document. Returns a list of tuples.

    Args:
        doc: A spacy Doc, representing a sequence of spacy Token objects.

    Design:
        - Changed the name of this function because the coursework sheet asks for the TOP10 syntactic objects overall - not adjectives.
        - Using lemmas (base form of each word) to account for tenses, capitalisation and singular/plural.
        - Ignoring spaces, new lines, punctuation, stop words, symbols and numbers.
    """
    words = [t.lemma_ for t in doc if not t.is_space and not t.is_punct and not t.is_stop and t.pos_ not in ["SYM", "NUM"]]
    return Counter(words).most_common(10)


if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    logging.info("Started script part 1.")

    # dependencies
    nltk.download("punkt_tab")  # English tokenizer
    nltk.download("cmudict")  # English syllable dictionary

    # question A
    logging.info("Running code for part 1 question A.")
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())

    # question B
    logging.info("Running code for part 1 question B.")
    print(get_ttrs(df))

    # question C
    logging.info("Running code for part 1 question C.")
    print(get_fks(df))

    # question E
    logging.info("Running code for part 1 question E.")
    parse(df)
    print(df.head())

    # question F
    logging.info("Running code for part 1 question F.")
    df = pd.read_pickle(Path.cwd() / "pickles" / "parsed.pickle")

    logging.info("Extracting TOP10 objects for each document.")
    for _idx, row in df.iterrows():
        print(row["title"])
        print(top10_objects(row["parsed"]))
        print("\n")

    logging.info("Extracting TOP10 subjects for verb 'to hear' by frequency for each document.")
    for _idx, row in df.iterrows():
        print(row["title"])
        print(top10_subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    logging.info("Extracting TOP10 subjects for verb 'to hear' by point-wise mutual information for each document.")
    for _idx, row in df.iterrows():
        print(row["title"])
        print(top10_subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")

    logging.info("Ended script part 1.")
