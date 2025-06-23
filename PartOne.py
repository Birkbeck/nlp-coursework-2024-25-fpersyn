#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

from pathlib import Path
import string
import re

import pandas as pd
import spacy
import nltk


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


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    pass


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


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    # dependencies
    nltk.download("punkt_tab")  # English tokenizer
    nltk.download("cmudict")  # English syllable dictionary

    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    # parse(df)
    # print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    # df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

