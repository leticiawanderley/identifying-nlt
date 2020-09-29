import torch
from typing import Dict, List
import pandas as pd

from constant import GROUND_TRUTH, MODEL_LABEL
from rnn import RNN
from rnn_pipeline import test_datapoint, train_rnn_model
from rnn_data_processing import get_all_tags, setup_data
from utils import setup_train_test_data
from visualization_functions import confusion_matrix


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
                              rnn_setup, n_hidden, learning_rate,
                              saved_model_path, 'all_losses_predict_nlt')
    else:
        rnn = RNN(len(all_tags), n_hidden, len(categories), rnn_setup)
        rnn.load_state_dict(torch.load(saved_model_path))
        rnn.eval()
    results_file = test_nlt_rnn(test_data, rnn, categories, all_tags)
    confusion_matrix(results_file, GROUND_TRUTH, MODEL_LABEL,
                     'confusion_matrix_predict_nlt')


if __name__ == "__main__":
    n_hidden = 128
    saved_model_path = 'saved_prediction_model.pth'
    predict_nlt(n_hidden, saved_model_path)
