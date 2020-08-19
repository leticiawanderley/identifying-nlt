# RNN pipeline

This implementation was based on the PyTorch tutorial [NLP From Scratch: Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html).

## Pre-processing training data
RNNs' most basic inputs are the representations of each individual unit in a sequence as a tensor, that is a one-hot vector as long as the amount of distinct units possible. In these experiments the unit tensors represent tags from the [Universal Dependencies tagset](https://universaldependencies.org/u/pos/index.html). As each language uses a subset of this tagset, we merge the tags used by the two languages under analysis, English and Chinese, and derive their one-hot vectors from that merged subset. To find out which tags are used we rely on preprocessed tag counts, computed from the corpora in each language.

The RNN model takes tag sequences as input. These sequences are represented as an array of one-hot tag vectors. The length of the array is the length of the tag sequence being represented.

## Training
The RNN model is trained with 256 hidden layers and it can use differents sets of learning rate, loss and activation functions. One of them is [Negative Log Likehood](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) as the loss function, [LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax) as the activation function, and a learning rate of 0.0001. Another option is using a learning rate of 0.0002 and [Binary Cross Entropy with Logits Loss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss), a function that combines the Binary Cross Entropy loss function with a Sigmoid layer and is more stable than using these two functions separately.

The network sees each training datapoint once. Its inputs are tag sequences, derived from the POS tagged sentences written in the languages analysed, and a flag that represents the language these sentences were written on. The RNN learns to predict these language flags from the tag sequences.

The trained RNN works as a native language identification (nli) model. It is trained to recognize the language a sentence was written on from the sentence's part-of-speech tag sequence. We hope the model learns both languages structures and is able to differentiate between the languages from this knowledge.

## Testing
### Test datasets
The test dataset for this model is [FCE dataset](https://www.aclweb.org/anthology/P11-1019/), a collection of error annotated essays written by learners who were sitting the FCE exam.

### Procedure

### Analysis

## Addendum