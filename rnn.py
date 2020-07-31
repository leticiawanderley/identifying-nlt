import math
import time
import torch
import torch.nn as nn

from rnn_data_preprocessing import get_all_tags, read_data
from rnn_helper_functions import category_from_output, random_training_example
from rnn_visualization_functions import confusion_matrix, losses


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


def train(category_tensor, sentence_tensor, learning_rate, criterion):
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    for i in range(sentence_tensor.size()[0]):
        output, hidden = rnn(sentence_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def evaluate(sentence_tensor):
    hidden = rnn.init_hidden()

    for i in range(sentence_tensor.size()[0]):
        output, hidden = rnn(sentence_tensor[i], hidden)

    return output


data, all_categories = read_data(
                        ['data/training data/tagged_sentences_1000sents.csv'],
                        ['en_ud', 'es_ud'])

all_tags = get_all_tags(['data/training data/es_ud_0_vocab.csv',
                         'data/training data/en_ud_0_vocab.csv'])

n_tags = len(all_tags)
n_categories = 2
n_hidden = 128
rnn = RNN(n_tags, n_hidden, n_categories)
criterion = nn.NLLLoss()
n_iters = 100000
print_every = 5000
plot_every = 1000
current_loss = 0
all_losses = []


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()
learning_rate = 0.005
for iter in range(1, n_iters + 1):
    category, sentence, category_tensor, sentence_tensor = \
        random_training_example(all_categories, data, n_tags, all_tags)
    output, loss = train(category_tensor, sentence_tensor,
                         learning_rate, criterion)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = category_from_output(output, all_categories)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100,
                                                time_since(start), loss,
                                                sentence, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, sentence, category_tensor, sentence_tensor = \
        random_training_example(all_categories, data, n_tags, all_tags)
    output = evaluate(sentence_tensor)
    guess, guess_i = category_from_output(output, all_categories)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

losses(all_losses, 'testesssss.png')
confusion_matrix(confusion, all_categories, 'confmat.png')
