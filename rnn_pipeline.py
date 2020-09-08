import time
from typing import Dict, List
import torch
import pandas as pd

from constant import GROUND_TRUTH, MODEL_LABEL
from rnn import RNN
from rnn_data_processing import category_from_output, Data, get_all_tags, \
                                read_data, sequence_to_tensor, setup_data
from utils import get_structural_errors, time_since, power_of_ten_value, \
                  setup_train_test_data
from visualization_functions import confusion_matrix, losses


def run_training(rnn: RNN, data: Data, categories: List,
                 learning_rate: float, output_model: str) -> List[float]:
    """Train RNN model printing training example and storing loss
    at every 5% and 1% of the data size , i.e., iteration, respectively.
    :param rnn: RNN object
    :param data: training data
    :param categories: output labels
    :param learning_rate: RNN learning rate
    :param output_model: path to save model in
    :return: stored losses
    """
    n_iters = data.size
    print_every = power_of_ten_value(n_iters * 0.05)
    plot_every = power_of_ten_value(n_iters * 0.01)
    start = time.time()
    current_loss = 0
    all_losses = []
    for iteration in range(1, n_iters + 1):
        category, sequence, category_tensor, sequence_tensor = \
            data.random_training_datapoint()
        output, loss = rnn.train_iteration(category_tensor, sequence_tensor,
                                           learning_rate)
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


def train_rnn_model(data: Dict[str, List[List[str]]],
                    categories: List, all_tags: List[str],
                    setup: str, learning_rate: float,
                    output_model: str, output_figure: str) -> RNN:
    """Train RNN model and create a losses visualization.
    :param data: training data
    :param categories: output labels
    :param setup: RNN parameters setup
    :param all_tags: all sequence tags possible
    :param learning_rate: RNN learning rate
    :param output_model: path to save model in
    :param output_figure: path to create losses figure in
    :return: trained RNN model
    """
    data = Data(data, all_tags, setup)
    rnn = RNN(len(all_tags), n_hidden, len(categories), setup)
    all_losses = run_training(rnn, data, categories, learning_rate,
                              output_model)
    losses(all_losses, output_figure)
    return rnn


def test_datapoint(rnn: RNN, datapoint: List[str], categories: List,
                   all_tags: List[str]):
    """Put datapoint through RNN model and return the model's guess.
    :param rnn: RNN model
    :param datapoint: test datapoint
    :param categories: output labels
    :param all_tags: all sequence tags possible
    :return: category guessed by the model
    """
    sequence_tensor = sequence_to_tensor(datapoint,
                                         len(all_tags), all_tags)
    output = rnn.evaluate(sequence_tensor)
    guess, guess_i = category_from_output(output, categories)
    return guess


def test_nli_rnn(test_dataset_file: str, rnn: RNN, categories: List[str],
                 all_tags: List[str]) -> str:
    """Test RNN model that classifies error tag sequences as negative language
    transfer through Native Language Identification, guessing the author's L1
    based on the error tag sequence.
    :param test_dataset_file: test dataset file path
    :param rnn: RNN model
    :param categories: output labels
    :param all_tags: all sequence tags possible
    :return: test results file name
    """
    test_df = pd.read_csv(test_dataset_file)
    nlt = []
    results = []
    structural_errors = get_structural_errors()
    for index, row in test_df.iterrows():
        if row['error_type'] == '_' or \
            row['error_type'] in structural_errors and \
                isinstance(row['incorrect_trigram_ud'], str):
            guess = test_datapoint(rnn, row['incorrect_trigram_ud'].split(),
                                   categories, all_tags)
            is_nlt = guess == 'zhs_ud'
            is_guess_correct = is_nlt == row[GROUND_TRUTH]
            nlt.append(is_nlt)
            results.append(is_guess_correct)
        else:
            nlt.append('')
            results.append('')
    test_df[MODEL_LABEL] = nlt
    test_df['result'] = results
    print(test_df.groupby(['result']).size().reset_index(name='count'))
    output_filename = 'data/results_chinese_annotated_errors_rnn_bce_3_cedar_more_data.csv'
    test_df.to_csv(output_filename)
    return output_filename


def nli(vocab_datasets: List[str], train_datasets: List[str],
        test_dataset_file: str, categories: List[str], n_hidden: int,
        saved_model_path: str, train_new_model=True):
    """Train and test RNN model that classifies error tag sequences as
    negative language transfer through Native Language Identification,
    guessing the author's native language based on the error tag sequence.
    :param vocab_datasets: list of files containing each language existing tags
    :param train_datasets: list of training data files
    :param test_dataset_file: test dataset file path
    :param categories: output labels
    :param n_hidden: number of hidden layers
    :param saved_model_path: path containing the model
    :param train_new_model: whether to train a new model or use the one saved
    """
    rnn_setup = 'BCEwithLL'
    all_tags = get_all_tags(vocab_datasets)
    if train_new_model:
        train_data = read_data(train_datasets, categories)
        learning_rate = 0.0001
        rnn = train_rnn_model(train_data, categories, all_tags,
                              rnn_setup, learning_rate,
                              saved_model_path, 'all_losses_zhs_en_bce_cedar_more_data.png')
    else:
        rnn = RNN(len(all_tags), n_hidden, len(categories), rnn_setup)
        rnn.load_state_dict(torch.load(saved_model_path))
        rnn.eval()
    results_file = test_nli_rnn(test_dataset_file, rnn,
                                categories, all_tags)
    confusion_matrix(results_file, GROUND_TRUTH, MODEL_LABEL,
                     'confusion_matrix_zhs_en_rnn_bce_3_cedar_more_data.png')


def test_nlt_rnn(test_data: pd.DataFrame, rnn: RNN, categories: List[bool],
                 all_tags: List[str]) -> str:
    """Test RNN model that classifies error tag sequences as negative language
    transfer or not.
    :param test_data: test dataset
    :param rnn: RNN model
    :param categories: output labels
    :param all_tags: all sequence tags possible
    :return: test results file name
    """
    nlt = []
    results = []
    for index, row in test_data.iterrows():
        guess = test_datapoint(rnn, row['penn_quadgram'].split(),
                               categories, all_tags)
        nlt.append(guess)
        results.append(guess == row[GROUND_TRUTH])
    test_data[MODEL_LABEL] = nlt
    test_data['result'] = results
    output_filename = 'data/results_predict_nlt_annotated_FCE.csv'
    print(test_data.groupby(['result']).size().reset_index(name='count'))
    test_data.to_csv(output_filename)
    return output_filename


def predict_nlt(n_hidden, saved_model_path, train_new_model=True):
    """Train and test RNN model that classifies error tag sequences as
    negative language transfer or not.
    :param n_hidden: number of hidden layers
    :param saved_model_path: path containing the model
    :param train_new_model: whether to train a new model or use the one saved
    """
    train_data, test_data = setup_train_test_data(
                                'data/testing data/annotated_FCE/' +
                                'chinese_annotated_errors.csv', 0.1,
                                GROUND_TRUTH)
    all_tags = get_all_tags(
                    ['data/training data/penn_quadgram_0_vocab.csv'])
    columns = {True: True, False: False}
    categories = list(columns.keys())
    rnn_setup = 'NLLoss'
    if train_new_model:
        data_dict = setup_data(train_data, columns, 'penn_quadgram',
                               GROUND_TRUTH)
        learning_rate = 0.15
        rnn = train_rnn_model(data_dict, categories, all_tags,
                              rnn_setup, learning_rate,
                              saved_model_path, 'all_losses_predict_nlt.png')
    else:
        rnn = RNN(len(all_tags), n_hidden, len(categories), rnn_setup)
        rnn.load_state_dict(torch.load(saved_model_path))
        rnn.eval()
    results_file = test_nlt_rnn(test_data, rnn, categories, all_tags)
    confusion_matrix(results_file, GROUND_TRUTH, MODEL_LABEL,
                     'confusion_matrix_predict_nlt.png')


if __name__ == "__main__":
    train_nli = False
    if train_nli:
        vocab_datasets = [
            'data/training data/chinese-english_vocabs/zhs_ud_0_vocab.csv',
            'data/training data/chinese-english_vocabs/en_ud_0_vocab.csv']
        train_datasets = [
                'data/training data/tagged_un_en-zh.csv',
                'data/training data/tagged_wmt-news_en-zh.csv',
                'data/training data/tagged_globalvoices_sentences.csv']
        test_dataset = 'data/testing data/annotated_FCE/' + \
                       'chinese_annotated_errors.csv'
        categories = ['en_ud', 'zhs_ud']
        n_hidden = 256
        saved_model_path = './saved_model_zhs_en_bce_cedar_more_data.pth'
        nli(vocab_datasets, train_datasets, test_dataset,
            categories, n_hidden, saved_model_path, True)
    else:
        n_hidden = 128
        saved_model_path = 'saved_prediction_model.pth'
        predict_nlt(n_hidden, saved_model_path)
