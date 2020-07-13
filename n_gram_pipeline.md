## N-gram pipeline

### Part-of-speech tagging
The sentences in the English and Spanish corpora were part-of-speech tagged using spaCy's POS tagging pipeline.
For English sentences there were extracted Universal Dependencies POS tags as well as the Penn Treebank POS tags.
They are refered to as "poss" and "tags", respectively.

The Spanish sentences were parsed using Universal Dependencies POS tags as well as AnCora POS tags. The later ones were then mapped to Penn Treebank POS tags using the mapping in [data/tags/spacy_spanish_tags_.csv](data/tags/spacy_spanish_tags_.csv). This mapping aims to enable the use of a more detailed shared tagset.

### Pre-processing training data
To speed up the training process, the POS tagged sentences were broken down into POS sequences of one, two, and three tags (unigrams, bigrams, and trigams). The POS sequences were then counted and stored in Python dictionaries along with their respective counts.

This step creates look-up tables for POS n-gram counts. With it, the n-gram models simply need to perform a key search operation to retrieve the amount of times a given POS sequence occurs in the corpora.

### Training
The n-gram models are trained using the algorithm described in [Jurafsky and Martin, 2019](https://web.stanford.edu/~jurafsky/slp3/3.pdf).

#### Out-of-vocabulary tags
A tag vocabulary is created to remove with low frequency and unknown tokens that could hinder the model performance. This vocabulary contains tags that occur more than 0.005% of the time in the tagged corpora, tags that are less frequent that that are replaced by an especial out-of-vocabulary token.

#### Smoothing techniques
There were implemented two smoothing techniques, add-one (or Laplace) and [deleted interpolation](https://web.stanford.edu/~jurafsky/slp3/8.pdf), apart from the unsmoothed option.

### Testing
