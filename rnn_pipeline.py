import time
import torch
import torch.nn as nn

from rnn import RNN
from rnn_data_preprocessing import get_all_tags, read_data
from rnn_helper_functions import category_from_output, random_training_example
from rnn_visualization_functions import confusion_matrix, losses
from utils import time_since


def main():
    data, all_categories = read_data(
                            ['data/training data/tagged_sentences_1000sents.csv'],
                            ['en_ud', 'es_ud'])

    all_tags = get_all_tags(['data/training data/es_ud_0_vocab.csv',
                             'data/training data/en_ud_0_vocab.csv'])
    n_hidden = 128
    rnn = RNN(len(all_tags), n_hidden, len(all_categories))
    n_iters = 100000
    print_every = 5000
    plot_every = 1000
    all_losses = run_training(rnn, data, all_categories, all_tags,
                              n_iters, print_every, plot_every)
    losses(all_losses, 'all_losses.png')
    confusion = build_confusion_data(rnn, all_categories, all_tags, data)
    confusion_matrix(confusion, all_categories, 'confusion_matrix.png')


def run_training(rnn, data, all_categories, all_tags, n_iters,
                 print_every, plot_every):
    start = time.time()
    n_tags = len(all_tags)
    learning_rate = 0.005
    criterion = nn.NLLLoss()
    current_loss = 0
    all_losses = []
    for iter in range(1, n_iters + 1):
        category, sentence, category_tensor, sentence_tensor = \
            random_training_example(all_categories, data, n_tags, all_tags)
        output, loss = rnn.train(category_tensor, sentence_tensor,
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
    return all_losses


def build_confusion_data(rnn, all_categories, all_tags, data):
    n_categories = len(all_categories)
    n_tags = len(all_tags)
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, sentence, category_tensor, sentence_tensor = \
            random_training_example(all_categories, data, n_tags, all_tags)
        output = rnn.evaluate(sentence_tensor)
        guess, guess_i = category_from_output(output, all_categories)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    return confusion


main()
