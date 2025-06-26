[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/8qgh5WxD)

# nlp-cw-template25
template for NLP module coursework

## Name

Fred Persyn

## Academic Declaration

“I have read and understood the sections of plagiarism in the College Policy
on assessment offences and confirm that the work is my own, with the work
of others clearly acknowledged. I give my permission to submit my report
to the plagiarism testing database that the College is using and test it using
plagiarism detection software, search engines or meta-searching software.”

## Part1 Question D

The Flesch-Kincaid Reading grade score is not a valid, robust or reliable estimator when:

* When dealing with a niche corpus or very technical vocabulary which can result in an inflated number of syllables per word.
* When documents or sentences are very short (e.g. tweets, short reviews, mobile text messages, ...) which will suppress the average sentence length.

## Part2 Question E

In my custom tokenizer, I've opted to parse the texts using spacy to subsequently apply more granual token selection.

I've focused on only extracting content tokens: ignoring spaces, punctuation, stop words and closed-class words (achieved via part-of-speech tagging).

To avoid word variations (e.g. capitalisation, tenses, singular vs plural), I've also opted to only extract the lemma for each word – it's base form.

Unfortunately, the current implementation does not improve classification performance at all compared to simple question D features (uni-/bi-/trigrams). The parsing also adds a considerable processing overhead (15+ min) slowing down iteration.

What I'd like to explore next:
* Implementing feature selection (of the TF-IDF vectors). For example, I'd expect a smaller set of features could be sufficient. For example, for the SNP I'd expect words such as "Scotland" or "independence" to carry high weights. (although technically part of a tokenizer)
* Increasing the min document frequency required (default: 1). Political speeches typically focus on repeated use of a set of keywords. This may make it more easy to ignore content that isn't the main focus of a speech. (although technically part of a tokenizer)
* Selecting words based on a polarity score.
