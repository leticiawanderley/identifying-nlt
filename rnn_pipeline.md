# RNN pipeline

This implementation was based on the PyTorch tutorial [NLP From Scratch: Classifying Names with a Character-Level RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html).

## Pre-processing training data
RNNs' most basic inputs are the representations of each individual unit in a sequence as a tensor, that is a one-hot vector as long as the amount of distinct units possible. In these experiments the unit tensors represent tags from the [Universal Dependencies tagset](https://universaldependencies.org/u/pos/index.html). As each language uses a subset of this tagset, we merge the tags used by the two languages under analysis, English and Chinese, and derive their one-hot vectors from that merged subset. To find out which tags are used we rely on preprocessed tag counts, computed from the corpora in each language.

The RNN model takes tag sequences as input. These sequences are represented as an array of one-hot tag vectors. The length of the array is the length of the tag sequence being represented.

## Training
The RNN model is trained with 256 hidden layers and it can use differents sets of learning rate, loss and activation functions. One of them is [Negative Log Likehood](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) as the loss function, [LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax) as the activation function, and a learning rate of 0.0001. Another option is using a learning rate of 0.0002 and [Binary Cross Entropy with Logits Loss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss), a function that combines the Binary Cross Entropy loss function with a Sigmoid layer and is more stable than using these two functions separately.

The network sees each training datapoint once. Its inputs are tag sequences, derived from the POS tagged sentences written in the languages analysed, and a flag that represents the language these sentences were written on. The RNN learns to predict these language flags from the tag sequences. The trained RNN works as a native language identification (nli) model. It is trained to recognize the language a sentence was written on from the sentence's part-of-speech tag sequence. We expect the model to learn both languages structures and to able to differentiate between the languages from this knowledge.

As a result of the training process the network's losses curve are plot in a figure.

## Testing
### Test datasets
The test dataset for this model is [FCE dataset](https://www.aclweb.org/anthology/P11-1019/), a collection of error annotated essays written by learners who were sitting the FCE exam. The errors in this dataset are annotated with information about whether they could be related to negative language transfer.

### Procedure
The main goal of this approach is to create a model that detects when a writing error could be caused by negative language transfer, the misguided influence of a learner's native language grammar rules in their writing in a foreign language. The trained RNN model takes in the sequence of tags surrounding and containing the incorrect utterance. It outputs a language flag indicating to which of the languages the tag sequence is more similar to.

### Analysis
If the model's output matches the learner's native language, the error is classified as being negative language transfer related. If the model outputs 'English', the error is classified as **not** transfer related. These classification results are compared to the annotated negative language transfer information. From this comparison it is possible to analyse the model's performance, as we can compare the model's results with the gold annotations in the dataset. These test results are also aggregated into a confunsion matrix.

## Addendum
Yet another RNN model can be trained using the FCE annotated dataset as both train and test data. This model takes the error tag sequence and error type as input and outputs whether that error is related to language transfer or not. It tries to predict the negative language transfer annotation from the writing error itself.