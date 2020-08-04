import time
import torch
import torch.nn as nn

from rnn import RNN
from rnn_data_preprocessing import get_all_tags, read_data
from rnn_helper_functions import category_from_output, Data
from rnn_visualization_functions import confusion_matrix, losses
from utils import time_since


def main():
    raw_data, all_categories = read_data(
                            ['data/training data/tagged_sentences_1000sents.csv'],
                            ['en_ud', 'es_ud'])
    
    all_tags = get_all_tags(['data/training data/es_ud_0_vocab.csv',
                             'data/training data/en_ud_0_vocab.csv'])
    data = Data(raw_data, all_tags)
    n_hidden = 128
    rnn = RNN(len(all_tags), n_hidden, len(all_categories))
    n_iters = data.size
    print_every = 50
    plot_every = 10
    all_losses = run_training(rnn, data, all_categories,
                              n_iters, print_every, plot_every)
    losses(all_losses, 'all_losses.png')
    data = Data(raw_data, all_tags)
    confusion = build_confusion_data(rnn, all_categories, data)
    confusion_matrix(confusion, all_categories, 'confusion_matrix.png')


def run_training(rnn, data, all_categories, n_iters,
                 print_every, plot_every):
    start = time.time()
    learning_rate = 0.005
    criterion = nn.NLLLoss()
    current_loss = 0
    all_losses = []
    for iter in range(1, n_iters + 1):
        category, sentence, category_tensor, sentence_tensor = \
            data.random_training_datapoint()
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


def build_confusion_data(rnn, all_categories, data):
    n_categories = len(all_categories)
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = data.size

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, sentence, category_tensor, sentence_tensor = \
            data.random_training_datapoint()
        output = rnn.evaluate(sentence_tensor)
        guess, guess_i = category_from_output(output, all_categories)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    return confusion


main()
