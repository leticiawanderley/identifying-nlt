# N-gram pipeline

## Part-of-speech tagging
The sentences in the English and Spanish corpora were part-of-speech tagged using spaCy's POS tagging pipeline.
For English sentences there were extracted Universal Dependencies POS tags as well as the Penn Treebank POS tags.
They are refered to as "ud" and "penn", respectively.

The Spanish sentences were parsed using Universal Dependencies POS tags as well as AnCora POS tags. The later ones were then mapped to Penn Treebank POS tags using the mapping in [data/tags/spacy_spanish_tags_.csv](data/tags/spacy_spanish_tags_.csv). This mapping aims to enable the use of a more detailed shared tagset.

## Pre-processing training data
To speed up the training process, the POS tagged sentences were broken down into POS sequences of one, two, and three tags (unigrams, bigrams, and trigams). These POS sequences were then gruped and counted so they could be stored in Python dictionaries along with their respective counts.

This step creates look-up tables for POS n-gram counts. With it, the n-gram models simply need to perform a key search operation to retrieve the amount of times a given POS sequence occurs in the corpora.

Since the sentences were tagged using two different tagsets there are two different collections of look-up tables, one for the UD tags and one for the Penn Treebank tags.

## Training
The n-gram models are trained using the algorithm described in [Jurafsky and Martin, 2019](https://web.stanford.edu/~jurafsky/slp3/3.pdf).

### Out-of-vocabulary tags
A tag vocabulary is created to filter out with low frequency and unknown tags that could hinder the model performance. This vocabulary contains tags that occur more than 0.005% of the time in the tagged corpora, tags that are less frequent that that are replaced by an especial out-of-vocabulary token.

### Smoothing techniques
There were implemented two smoothing techniques, add-one (or Laplace) and [deleted interpolation](https://web.stanford.edu/~jurafsky/slp3/8.pdf), apart from the unsmoothed option.

## Testing
### Test datasets
There are two datasets that can be used to test the models. The first one was extracted from the book [Learner English - A teacher's guide to interference and other problems](https://books.google.ca/books/about/Learner_English.html?id=6UIuWj9fQfQC), which is a collection of common errors made by non-native English speakers grouped by the learner's first languages. The second one is the [FCE dataset](https://www.aclweb.org/anthology/P11-1019/), a collection of error annotated essays written by learners who were sitting the FCE exam.

### Procedure
Datapoints in both datasets were annotated with error types, and the specific error sequences were highlighted in a separate column. These three-tag sequences are the input to the language models. Only errors with structural errors types are feed as input to the language models. The structural errors types can be found in [data/error_type_meaning.csv](data/error_type_meaning.csv). Each language model computes the likehood that these sequences belong to the language it represents and write this likehood into the error row.

### Analysis
The likehoods computed by the language models tell us how common a sequence is in that language. We analyse the performance of this approach by calculating whether known negative language transfer effects are captured by the difference in the models predictions. If a sequence is more frequent in the learner's native language model than in the English language model, we tag it as a negative language transfer error. We measure this approach's performance by computing how many expert annotated structural negative trasnfer errors are correctly classified as negative language transfer errors by the language models combination.