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
* When documents or sentences are very short (e.g. tweets, short reviews, telegrams, advertising copy, ...) which will suppress the average sentence length.

## Part2 Question E

In my custom tokenizer, I've opted to parse the texts using spacy to subsequently apply more granual token selection.

I've focused on only extracting content tokens: ignoring spaces, punctuation, stop words and closed-class words (achieved via part-of-speech tagging).

To avoid word variations (e.g. capitalisation, tenses, singular vs plural), I've also opted to only extract the lemma for each word – it's base form.

Unfortunately, the current implementation does not improve classification performance at all compared to simple question D features (uni-/bi-/trigrams). The parsing also adds a considerable processing overhead (15+ min) slowing down iteration.

Further iteration - using feature selection:
Although not technically part of a tokenizer, I've implemented basic feature selection – selecting the top500 vectors based on a chi2 test with the target variable. My motivation for choosing feature selection is that many words should carry significant signal for their respective party. For example, I'd expect words such as "Scotland" or "independence" to carry high weights for the SNP. This improves the performance of the random forest classifier considerably (especially accuracy and F1) but doesn't add any incremental performance for the linear SVM.

Further iteration - using minimum document frequency:
I've increased the minimum document word count to 2 (default: 1). Political speeches typically focus on repeated use of a set of keywords. This may make it more easy to ignore content that isn't the main content of a political speech. Bizarelly, this improves F1 classification performance for all classes but the Liberal Democrats – for both models.

What I'd like to explore next:
* Explore alternative feature selection metrics (e.g. point-wise PMI, word polarity score, ...)
* Further experimentation with minimum document frequency
