# N-gram pipeline

## Part-of-speech tagging
The sentences in the English, Spanish, and Chinese corpora were part-of-speech tagged using spaCy's POS tagging pipeline.
For English sentences there were extracted Universal Dependencies POS tags as well as the Penn Treebank POS tags.
They are refered to as "ud" and "penn", respectively.

The Spanish sentences were parsed using Universal Dependencies POS tags as well as AnCora POS tags. The later ones were then mapped to Penn Treebank POS tags using the mapping in [data/tags/spacy_spanish_tags_.csv](data/tags/spacy_spanish_tags_.csv). This mapping aims to enable the use of a more detailed shared tagset. As in the English corpora, Universal Dependencies POS tags are refered to as "ud" and the mapped Penn Treebank POS tags are called "penn".

[SpaCy](https://spacy.io/usage/linguistic-features#pos-tagging) distinguishes between simple Universal Dependecies POS tags and more detailed language specific tags by refering to them as "POS" and "Tag", respectively. These two denomination can be mapped to our "ud" and "penn" categories.

## Pre-processing training data
The training data was pre-processed into `.arpa` files using [KenLM estimation](https://kheafield.com/code/kenlm/estimation/).

## Training
The models were trained using the n-gram language model implementation from [KenLM](https://github.com/kpu/kenlm).

### Tuning
The n-gram length used to training the models (n = 5) was selected through a process of parameter tuning. In this process, the language source prediction accuracy was computed as the mean accuracy of a 5-fold cross-validation evaluation using the training data.

## Testing
### Test datasets
There are two datasets that can be used to test the models. The first one was extracted from the book [Learner English - A teacher's guide to interference and other problems](https://books.google.ca/books/about/Learner_English.html?id=6UIuWj9fQfQC), which is a collection of common errors made by non-native English speakers grouped by the learner's first languages. The second one is the [FCE dataset](https://www.aclweb.org/anthology/P11-1019/), a collection of error annotated essays written by learners who were sitting the FCE exam.

### Procedure
Datapoints in both datasets were annotated with error types, and the specific error sequences were highlighted in a separate column. Three distinct windows were used to represent error sequences. Only errors with structural errors types are feed as input to the language models. The structural errors types can be found in [data/error_type_meaning.csv](data/error_type_meaning.csv). Each language model computes the likehood that these sequences belong to the language it represents and write this likehood into the error row.

### Analysis
The likehoods computed by the language models tell us how common a sequence is in that language. We analyse the performance of this approach by calculating whether known negative language transfer effects are captured by the difference in the models predictions. If a sequence is more frequent in the learner's native language model than in the English language model, we tag it as a negative language transfer error. We measure this approach's performance by computing how many expert annotated structural negative transfer errors are correctly classified as negative language transfer errors by the comparison of the language models results.