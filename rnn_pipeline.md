# RNN pipeline

This implementation was based on the PyTorch tutorial [NLP From Scratch: Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html).

## Pre-processing training data
RNNs' most basic inputs are the representations of each individual unit in a sequence as a tensor, that is a one-hot vector as long as the amount of distinct units possible. In these experiments the unit tensors represent tags from the [Universal Dependencies tagset](https://universaldependencies.org/u/pos/index.html). As each language uses a subset of this tagset, we merge the tags used by the two languages in analysis, English and Chinese, and derive their one-hot vectors from that subset. To find out which tags are used we rely on preprocessed tag counts, computed from the corpora in each language.

The RNN model takes tag sequences as input. These sequences are represented as an array of one-hot tag vectors. The length of the array is the length of the tag sequence being represented.

## Training


## Testing
### Test datasets

### Procedure

### Analysis
