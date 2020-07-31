import math
import pandas as pd
import random
import time
import torch
import torch.nn as nn


def read_data(filenames, columns):
    data = {}
    for column in columns:
        data[column] = []
    for filename in filenames:
        df = pd.read_csv(filename)
        for c in columns:
            data[c] += df[c].to_list()
    return data, list(data.keys())


def get_all_tags(languages_tag_files):
    all_tags = {}
    for tag_file in languages_tag_files:
        df = pd.read_csv(tag_file, index_col=0)
        for tag in df['ngram'].to_list():
            if tag not in all_tags:
                all_tags[tag] = True
    return sorted(all_tags.keys())


def tag_to_index(tag, all_tags):
    return all_tags.index(tag)


def tag_to_tensor(tag, n_tags, all_tags):
    tensor = torch.zeros(1, n_tags)
    tensor[0][tag_to_index(tag, all_tags)] = 1
    return tensor


def sentence_to_tensor(sentence, n_tags, all_tags):
    sentence = sentence.split()
    tensor = torch.zeros(len(sentence), 1, n_tags)
    for li, tag in enumerate(sentence):
        tensor[li][0][tag_to_index(tag, all_tags)] = 1
    return tensor


data, all_categories = read_data(
                        ['data/training data/tagged_sentences_1000sents.csv'],
                        ['en_ud', 'es_ud'])

all_tags = get_all_tags(['data/training data/es_ud_0_vocab.csv',
                         'data/training data/en_ud_0_vocab.csv'])

n_tags = len(all_tags)


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


n_categories = 2
n_hidden = 128
rnn = RNN(n_tags, n_hidden, n_categories)


def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


input = sentence_to_tensor('DET NOUN VERB ADP NUM PROPN ', n_tags, all_tags)
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden)
print(output)

print(category_from_output(output, all_categories))


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]

def random_training_example(all_categories, data, n_tags, all_tags):
    category = random_choice(all_categories)
    sentence = random_choice(data[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    sentence_tensor = sentence_to_tensor(sentence, n_tags, all_tags)
    return category, sentence, category_tensor, sentence_tensor

criterion = nn.NLLLoss()

def train(category_tensor, sentence_tensor, learning_rate):
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


n_iters = 100000
print_every = 5000
plot_every = 1000
# Keep track of losses for plotting
current_loss = 0
all_losses = []


def timeSince(since):
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
                         learning_rate)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = category_from_output(output, all_categories)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100,
                                                timeSince(start), loss,
                                                sentence, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.savefig('testeeeee3.png')


# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(sentence_tensor):
    hidden = rnn.init_hidden()

    for i in range(sentence_tensor.size()[0]):
        output, hidden = rnn(sentence_tensor[i], hidden)

    return output

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

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

fig.savefig('confmat.png')
