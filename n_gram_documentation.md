## N-gram models

### Part-of-speech tagging
The sentences in the English and Spanish corpora were part-of-speech tagged using spaCy's POS tagging pipeline.
For English sentences there were extracted Universal Dependencies POS tags as well as the Penn Treebank POS tags.
They are refered to as "poss" and "tags", respectively.

The Spanish sentences were parsed into Universal Dependencies POS tags as well as AnCora POS tags. The later ones were then mapped to Penn Treebank POS tags using the mapping in [data/spaCy tags/spacy_spanish_tags_.csv](data/spaCy tags/spacy_spanish_tags_.csv). This mapping aims to enable the use of a more detailed shared tagset.

### Pre-processing training data
