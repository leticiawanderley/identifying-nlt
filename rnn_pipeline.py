import time
import torch
import torch.nn as nn

from rnn import RNN
from rnn_data_preprocessing import get_all_tags, read_data
from rnn_helper_functions import category_from_output, Data
from rnn_visualization_functions import confusion_matrix, losses
from utils import time_since


def main(train_new_model=True):
    all_tags = get_all_tags(['data/training data/es_ud_0_vocab.csv',
                             'data/training data/en_ud_0_vocab.csv'])
    categories = ['en_ud', 'es_ud']
    n_hidden = 256
    saved_model_path = './saved_model.pth'
    if train_new_model:
        training_data = read_data(
            ['data/training data/europarl/english_spanish_10_tagged.csv',
             'data/training data/europarl/english_spanish_11_tagged.csv',
             'data/training data/europarl/english_spanish_12_tagged.csv',
             'data/training data/europarl/english_spanish_13_tagged.csv',
             'data/training data/europarl/english_spanish_14_tagged.csv',
             'data/training data/europarl/english_spanish_15_tagged.csv',
             'data/training data/europarl/english_spanish_16_tagged.csv',
             'data/training data/europarl/english_spanish_17_tagged.csv',
             'data/training data/europarl/english_spanish_18_tagged.csv',
             'data/training data/europarl/english_spanish_19_tagged.csv',
             'data/training data/europarl/english_spanish_20_tagged.csv'],
            categories)
        data = Data(training_data, all_tags)
        rnn = RNN(len(all_tags), n_hidden, len(categories))
        n_iters = data.size
        print_every = 5000
        plot_every = 1000
        all_losses = run_training(rnn, data, categories,
                                  n_iters, print_every, plot_every)
        losses(all_losses, 'all_losses.png')
    test_data = read_data(
        ['data/training data/tagged_sentences_1000sents.csv'],
        categories)
    data = Data(test_data, all_tags)
    saved_rnn = RNN(len(all_tags), n_hidden, len(categories))
    saved_rnn.load_state_dict(torch.load(saved_model_path))
    saved_rnn.eval()
    confusion = build_confusion_data(saved_rnn, categories, data)
    confusion_matrix(confusion, categories, 'confusion_matrix.png')


def run_training(rnn, data, categories, n_iters,
                 print_every, plot_every):
    start = time.time()
    learning_rate = 0.0002
    criterion = nn.NLLLoss()
    current_loss = 0
    all_losses = []
    for iteration in range(1, n_iters + 1):
        category, sequence, category_tensor, sequence_tensor = \
            data.random_training_datapoint()
        output, loss = rnn.train_iteration(category_tensor, sequence_tensor,
                                           learning_rate, criterion)
        current_loss += loss

        # Print iteration number, loss, name and guess
        if iteration % print_every == 0:
            guess, guess_i = category_from_output(output, categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iteration,
                                                    iteration / n_iters * 100,
                                                    time_since(start), loss,
                                                    sequence, guess, correct))

        # Add current loss avg to list of losses
        if iteration % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    torch.save(rnn.state_dict(), './saved_model.pth')
    return all_losses


def build_confusion_data(rnn, categories, data):
    n_categories = len(categories)
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = data.size

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, sequence, category_tensor, sequence_tensor = \
            data.random_training_datapoint()
        output = rnn.evaluate(sequence_tensor)
        guess, guess_i = category_from_output(output, categories)
        category_i = categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    return confusion


main()
