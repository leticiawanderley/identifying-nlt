# RNN pipeline

This implementation was based on the PyTorch tutorial [NLP From Scratch: Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html).

## Pre-processing training data
RNNs' most basic inputs are the representations of each individual unit in a sequence as a tensor, that is a one-hot vector as long as the amount of distinct units possible. In these experiments the unit tensors represent tags from the [Universal Dependencies tagset](https://universaldependencies.org/u/pos/index.html). As each language uses a subset of this tagset, we merge the tags used by the two languages under analysis, English and Chinese, and derive their one-hot vectors from that merged subset. To find out which tags should be used, we relied on preprocessed tag counts computed from the corpora in each language.

The RNN model takes tag sequences as input. These sequences are represented as an array of one-hot tag vectors. The length of the array is the length of the tag sequence being represented.

## Training
The RNN model is trained for 10 epochs with Adam optimizaton. It has 16 hidden layers, learning rate = 0.0001, mini batch size = 1, and [Negative Log Likehood](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) loss function, and [LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax) as the activation function. These parameters were seleted through a process of parameter tuning.

The network sees each training datapoint once. Its inputs are tag sequences, derived from the POS tagged sentences written in the languages analysed, and a flag that represents the language these sentences were written on. The RNN learns to predict these language flags from the tag sequences. The trained RNN works as a source language identification model. It is trained to recognize the language a sentence was written on from the sentence's part-of-speech tag sequence. We expect the model to learn both languages structures and to able to differentiate between the languages from this knowledge.

As a result of the training process the network's losses curve are plot in a figure.

### Tuning
The learning rate, number of hidden units, mini batch size, and loss + activation functions used to train the RNN model were selected during tuning. The tuning procedure consisted in training RNN models with different parameter combinations on [80% of the training data](data/training_data/chinese_english_splits/train_split.csv), and evaluating their source language predicton accuracy on [the held-out 20% of the training data](data/training_data/chinese_english_splits/eval_split.csv). The parameter combination that yielded the best accuracy was selected to train the negative language transfer detection RNN model.

## Testing
### Test datasets
The test dataset for this model is [FCE dataset](https://www.aclweb.org/anthology/P11-1019/), a collection of error annotated essays written by learners who were sitting the FCE exam. The errors in this dataset are annotated with information about whether they could be related to negative language transfer.

### Procedure
The main goal of this approach is to create a model that detects when a writing error could be caused by negative language transfer, the misguided influence of a learner's native language grammar rules in their writing in a foreign language. The trained RNN model takes in the sequence of tags surrounding and containing the incorrect utterance. It outputs a language flag indicating to which of the languages the tag sequence is more similar to.

### Analysis
If the model's output matches the learner's native language, the error is classified as being negative language transfer related. If the model outputs 'English', the error is classified as **not** transfer related. These classification results are compared to the annotated negative language transfer information. From this comparison it is possible to analyse the model's performance, as we can compare the model's results with the gold standard annotations in the dataset. These test results are aggregated and represented as a confunsion matrix.