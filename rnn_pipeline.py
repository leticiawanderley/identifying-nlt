import argparse
from typing import Dict, List
import torch
import pandas as pd


from torch.utils.data import DataLoader
from constant import GROUND_TRUTH, MODEL_LABEL
from rnn import RNN
from rnn_data_processing import category_from_output, Dataset, get_all_tags, \
                                read_data, sequence_to_tensor
from visualization_functions import confusion_matrix, losses


def run_training(rnn: RNN, data_dict: Dict[str, List[List[str]]],
                 eval_data_dict: Dict[str, List[List[str]]],
                 categories: List, setup: str,
                 learning_rate: float, batch_size: int, output_model: str,
                 test_dataset_file: str = None, all_tags: List[str] = None,
                 test_column: str = None) -> List[float]:
    """Train RNN model
    :param rnn: RNN object
    :param data_dict: training data
    :param eval_data_dict: evaluation data split
    :param categories: output labels
    :param setup: RNN parameters setup
    :param learning_rate: RNN learning rate
    :param batch_size: Mini batch size
    :param output_model: path to save model in
    :param test_dataset_file: test dataset file path
    :param all_tags: possible sequence tags
    :param test_column: column that contains test sequences
    :return: stored losses and partial results
    """
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    all_losses = []
    test_results = {}
    data_loader = DataLoader(
        Dataset(data_dict, all_tags, setup), batch_size, shuffle=True)
    epochs = 10
    for e in range(epochs):
        for x, y, l in data_loader:
            output, loss = rnn.train_iteration(y, x, l,
                                               optimizer)
            all_losses.append(loss)
        test_set_accuracy = 0.0
        if eval_data_dict:
            test_set_accuracy = tuning_evaluation(rnn, eval_data_dict,
                                                  categories, all_tags)
        print('%.2f Accuracy on test set' % test_set_accuracy)
        if test_dataset_file:
            iter_results = test_iteration(test_dataset_file, rnn,
                                          categories, all_tags,
                                          test_column)
            iter_results.append(str(test_set_accuracy))
            test_results['100%,' + str(e)] = iter_results
    torch.save(rnn.state_dict(), output_model)
    return all_losses, test_results


def tuning_evaluation(rnn: RNN, test_data: Dict[str, List[str]],
                      categories: List, all_tags: List[str]) -> float:
    """Evaluate RNN with the tuning evaluation data split
    :param rnn: RNN object
    :param test_data: test dataset
    :param categories: output labels
    :param all_tags: possible sequence tags
    :return: model's accuracy
    """
    size = 0
    correct = 0
    for key in test_data:
        size += len(test_data[key])
        for sequence in test_data[key]:
            guess = test_datapoint(rnn, sequence, categories, all_tags)
            if guess == key:
                correct += 1
    return correct/size if size > 0 else 0


def train_rnn_model(data: Dict[str, List[List[str]]],
                    eval_data: Dict[str, List[List[str]]],
                    categories: List, all_tags: List[str],
                    setup: str, n_hidden: int, learning_rate: float,
                    batch_size: int, info: str,
                    test_dataset_file: str = None,
                    test_column: str = None) -> RNN:
    """Train RNN model and create a losses visualization.
    :param data: training data
    :param eval_data: model evaluation data
    :param categories: output labels
    :param setup: RNN parameters setup
    :param n_hidden: number of hidden units
    :param all_tags: possible sequence tags
    :param learning_rate: RNN learning rate
    :param batch_size: Mini batch size
    :param info: hyperparameter information
    :param test_dataset_file: test dataset file path
    :param test_column: column that contains test sequences
    :return: trained RNN model
    """
    rnn = RNN(len(all_tags), n_hidden, len(categories), setup)
    all_losses, test_results = run_training(rnn, data, eval_data,
                                            categories, setup,
                                            learning_rate,
                                            batch_size, info,
                                            test_dataset_file,
                                            all_tags, test_column)
    if test_dataset_file:
        f = open('partial_results.csv', 'a')
        for key in test_results:
            f.write(setup + ',' + str(n_hidden) + ',' +
                    str(learning_rate) + ',' + str(batch_size) +
                    ',' + str(key) + ',' +
                    ','.join(test_results[key]) + '\n')
        f.close()
    losses(all_losses, info)
    return rnn


def test_datapoint(rnn: RNN, datapoint: List[str], categories: List,
                   all_tags: List[str]):
    """Put datapoint through RNN model and return the model's guess.
    :param rnn: RNN model
    :param datapoint: test datapoint
    :param categories: output labels
    :param all_tags: possible sequence tags
    :return: category guessed by the model
    """
    sequence_tensor = sequence_to_tensor(datapoint,
                                         len(all_tags), all_tags,
                                         len(datapoint))
    output = rnn.evaluate(sequence_tensor, is_batch=False)
    guess, guess_i = category_from_output(output, categories)
    return guess


def test_iteration(test_dataset_file: str, rnn: RNN, categories: List[str],
                   all_tags: List[str], test_column: str) -> List[str]:
    """Test RNN partial model
    :param test_dataset_file: test dataset file path
    :param rnn: RNN model
    :param categories: output labels
    :param all_tags: possible sequence tags
    :param test_column: column that contains test sequences
    :return: true and false positives for both transfer and non-transfer errors
    """
    test_df = pd.read_csv(test_dataset_file)
    nlt_tp = 0
    nlt_fp = 0
    nlt_fn = 0
    nlt_tn = 0
    for index, row in test_df.iterrows():
        if row['error_type'] == '_' and \
           isinstance(row[test_column], str) and row[test_column] != ' ':
            guess = test_datapoint(rnn, row[test_column].split(),
                                   categories, all_tags)
            if row[GROUND_TRUTH]:
                if guess == 'zhs_ud':
                    nlt_tp += 1
                else:
                    nlt_fn += 1
            else:
                if guess == 'zhs_ud':
                    nlt_fp += 1
                else:
                    nlt_tn += 1
    return [str(nlt_tp), str(nlt_fp), str(nlt_tn), str(nlt_fn)]


def test_nli_rnn(test_dataset_file: str, rnn: RNN, categories: List[str],
                 all_tags: List[str], test_column: str, info: str,
                 correct_test_column: str = None) -> str:
    """Test RNN model that classifies error tag sequences as negative language
    transfer through Native Language Identification, guessing the author's L1
    based on the error tag sequence.
    :param test_dataset_file: test dataset file path
    :param rnn: RNN model
    :param categories: output labels
    :param all_tags: possible sequence tags
    :param test_column: column that contains test sequences
    :param info: hyperparameter information
    :return: test results file name
    """
    test_df = pd.read_csv(test_dataset_file, index_col=[0])
    nlt = []
    results = []
    for index, row in test_df.iterrows():
        if isinstance(row[test_column], str) and \
                row[test_column] != ' ':
            guess = test_datapoint(rnn, row[test_column].split(),
                                   categories, all_tags)
            is_nlt = guess == 'zhs_ud'
            is_guess_correct = is_nlt == row[GROUND_TRUTH]
            nlt.append(is_nlt)
            results.append(is_guess_correct)
        else:
            nlt.append(not row[GROUND_TRUTH])
            results.append(False)
    test_df[MODEL_LABEL] = nlt
    test_df['result'] = results
    print(test_df.groupby(['result']).size().reset_index(name='count'))
    output_filename = 'data/results/results_' + info + \
                      '_' + test_column + '.csv'
    test_df.to_csv(output_filename)
    return output_filename


def nli(vocab_datasets: List[str], train_datasets: List[str],
        eval_datasets: List[str],
        test_dataset_file: str, learning_rate: float,
        loss: str, batch_size: int, categories: List[str], n_hidden: int,
        test_column: str, info: str, train_new_model=True):
    """Train and test RNN model that classifies error tag sequences as
    negative language transfer through Native Language Identification,
    guessing the author's native language based on the error tag sequence.
    :param vocab_datasets: list of files containing each language existing tags
    :param train_datasets: list of training data files
    :param eval_datasets: list of evaluation data files
    :param test_dataset_file: test dataset file path
    :param learning_rate: initial learning rate
    :param loss: RNN's loss function
    :param batch_size: Mini batch size
    :param categories: output labels
    :param n_hidden: number of hidden layers
    :param test_column: column that contains test sequences
    :param info: hyperparameter information
    :param train_new_model: whether to train a new model or use the one saved
    """
    all_tags = get_all_tags(vocab_datasets)
    if train_new_model:
        data = read_data(train_datasets, categories)
        eval_data = None
        if eval_datasets:
            eval_data = read_data(eval_datasets, categories)
        rnn = train_rnn_model(data, eval_data, categories, all_tags,
                              loss, n_hidden, learning_rate, batch_size,
                              info, test_dataset_file,
                              test_column)
    else:
        rnn = RNN(len(all_tags), n_hidden, len(categories), loss)
        rnn.load_state_dict(torch.load(info))
        rnn.eval()
    results_file = test_nli_rnn(test_dataset_file, rnn, categories,
                                all_tags, test_column, info)
    confusion_matrix(results_file, GROUND_TRUTH, MODEL_LABEL, info)


def parse_arg_list():
    """Uses argparse to parse the required parameters"""
    parser = argparse.ArgumentParser(
                description='',
                formatter_class=argparse.RawTextHelpFormatter)
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument(
        '-lr', '--learning_rate', help='Initial learning rate')
    required_args.add_argument(
        '-nh', '--number_hidden', help='Number of hidden units')
    required_args.add_argument(
        '-tc', '--test_column', help='Test dataset column')
    required_args.add_argument(
        '-lo', '--loss', help='Loss function')
    required_args.add_argument(
        '-mb', '--mini_batch', help='Mini batch size')
    required_args.add_argument(
        '-f', '--input_file', help='File that contains input parameters')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    vocab_datasets = [
        'data/training_data/chinese_english_vocabs/zhs_ud_0_vocab.csv',
        'data/training_data/chinese_english_vocabs/en_ud_0_vocab.csv']
    train_datasets = [
        'data/training_data/chinese_english_splits/train_split.csv',
        'data/training_data/chinese_english_splits/eval_split.csv']
    eval_datasets = []
    test_dataset = 'data/test_data/zhs_structural_errors.csv'
    categories = ['en_ud', 'zhs_ud']
    args = parse_arg_list()
    if args.input_file:
        f = open(args.input_file, 'r').readlines()
        learning_rate = f[0].strip('\n')
        number_hidden = f[1].strip('\n')
        test_column = f[2].strip('\n')
        loss = f[3].strip('\n')
        batch_size = f[4].strip('\n')
    else:
        learning_rate = args.learning_rate
        number_hidden = args.number_hidden
        test_column = args.test_column
        loss = args.loss
        batch_size = args.mini_batch
    info = '_'.join(categories) + '_' + learning_rate + \
           '_' + number_hidden + '_' + loss + '_' + batch_size
    nli(vocab_datasets, train_datasets, eval_datasets,
        test_dataset, float(learning_rate), loss, int(batch_size),
        categories, int(number_hidden), test_column, info, False)
