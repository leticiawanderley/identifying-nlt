import time
import torch
import torch.nn as nn
import pandas as pd

from rnn import RNN
from rnn_data_preprocessing import get_all_tags, read_data, sequence_to_tensor
from rnn_helper_functions import category_from_output, Data
from rnn_visualization_functions import confusion_matrix, losses
from utils import get_structural_errors, time_since
from constant import ANNOTATED_FCE_FIELDS

def main(train_new_model=True):
    all_tags = get_all_tags(['data/training data/globalvoices_vocabs/' +
                             'zhs_ud_0_vocab.csv',
                             'data/training data/globalvoices_vocabs/' +
                             'en_ud_0_vocab.csv'])
    categories = ['en_ud', 'zhs_ud']
    n_hidden = 256
    saved_model_path = './saved_model_zhs_en_1.pth'
    if train_new_model:
        training_data = read_data(
            ['data/training data/tagged_globalvoices_sentences.csv'],
            categories)
        data = Data(training_data, all_tags)
        rnn = RNN(len(all_tags), n_hidden, len(categories))
        n_iters = data.size
        print_every = 5000
        plot_every = 1000
        all_losses = run_training(rnn, data, categories,
                                  n_iters, print_every, plot_every,
                                  saved_model_path)
        losses(all_losses, 'all_losses_zhs_es_1.png')
    test_data_dict = {}
    test_data = pd.read_csv('data/testing data/annotated_FCE/' +
                            'chinese_annotated_errors.csv')
    test_data_dict['zhs_ud'] = \
        test_data[test_data['Negative transfer?'] == True]\
        ['incorrect_trigram_ud'].to_list()
    test_data_dict['en_ud'] = \
        test_data[test_data['Negative transfer?'] == False]\
        ['incorrect_trigram_ud'].to_list()
    for cat in categories:
        for i in range(len(test_data_dict[cat])):
            test_data_dict[cat][i] = test_data_dict[cat][i].split()
    data = Data(test_data_dict, all_tags)
    saved_rnn = RNN(len(all_tags), n_hidden, len(categories))
    saved_rnn.load_state_dict(torch.load(saved_model_path))
    saved_rnn.eval()
    confusion = build_confusion_data(saved_rnn, categories, data)
    confusion_matrix(confusion, categories, 'confusion_matrix_zhs_en_1.png')
    test_annotated_fce(saved_rnn, categories, len(all_tags), all_tags)


def run_training(rnn, data, categories, n_iters,
                 print_every, plot_every, output_model):
    start = time.time()
    learning_rate = 0.0001
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
    torch.save(rnn.state_dict(), output_model)
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
    print(confusion)
    return confusion


def test_annotated_fce(rnn, categories, n_tags, all_tags):
    test_df = pd.read_csv('data/testing data/annotated_FCE/' +
                          'chinese_annotated_errors.csv')
    nlt = []
    results = []
    structural_errors = get_structural_errors()
    for index, row in test_df.iterrows():
        if row['error_type'] == '_' or row['error_type'] in structural_errors:
            sequence_tensor = sequence_to_tensor(
                                row['incorrect_trigram_ud'].split(),
                                n_tags, all_tags)
            output = rnn.evaluate(sequence_tensor)
            guess, guess_i = category_from_output(output, categories)
            is_nlt = guess == 'zhs_ud'
            is_guess_correct = is_nlt == row['Negative transfer?']
            nlt.append(is_nlt)
            results.append(is_guess_correct)
        else:
            nlt.append('')
            results.append('')
    test_df['nlt'] = nlt
    test_df['result'] = results
    print(test_df.groupby(['result']).size().reset_index(name='count'))
    test_df.to_csv('data/results_chinese_annotated_errors_rnn.csv')


main(False)
