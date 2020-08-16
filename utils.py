import csv
import math
import numpy as np
import pandas as pd
import time

from constant import CCONJ, CONJ, SPANISH
from sklearn.model_selection import train_test_split


def split_sentences(dataset, end_of_sentence=False):
    """Create list of tags from each of the datasets' rows
    appending two end of sentence markers to the end of the lists."""
    clean_dataset = []
    for i in range(len(dataset)):
        if type(dataset[i]) == str:
            row = dataset[i].split()
            if end_of_sentence:
                row += ['_', '_']
            clean_dataset.append(row)
    return clean_dataset


def extract_vocabs(dataset, n):
    """For each gram from unigram to n-gram, create a dictionary in which
    the keys are the n-grams that exist on the dataset
    and the values are their occurence counts."""
    vocabs = {}
    for length in range(n):
        vocabs[length] = {}
    for sent in dataset:
        for i in range(len(sent)):
            for j in range(n):
                right_index = i + j + 1
                if right_index <= len(sent):
                    key = ' '.join(sent[i:right_index])
                    if key not in vocabs[j].keys():
                        vocabs[j][key] = 0
                    vocabs[j][key] += 1
    return vocabs


def get_count(tags, vocabs):
    """Retrive number of occurences of tag sequence
    from the pre-processed n-gram vocabularies."""
    vocab = vocabs[len(tags) - 1]
    tags_key = ' '.join(tags)
    return vocab[tags_key] if tags_key in vocab.keys() else 0


def tags_mapping(filename):
    """Read csv tags mapping and convert it to a dictionary."""
    mapping_dict = {}
    with open(filename, newline='') as csvfile:
        mapping = csv.reader(csvfile, delimiter=',')
        for row in mapping:
            mapping_dict[row[0]] = row[1]
    return mapping_dict


def unpack_ud_and_penn_tags(series):
    """Unpack series of parsed ud and penn treebank tags into two lists."""
    ud = []
    penn = []
    for row in series.to_list():
        ud.append(row[0])
        penn.append(row[1])
    return ud, penn


def process_tags(input_filename, output_filename):
    """Create csv file from txt file in which the values are
    separated by commas."""
    f = open(input_filename, "r")
    lines = f.readlines()
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for line in lines:
            tags = line.split(',')
            for tag in tags:
                csvwriter.writerow([tag.strip()])


def save_vocabs_to_csv(vocabs, lang, affix):
    """Save ngram counts to a csv file."""
    for n in vocabs.keys():
        df = pd.DataFrame(vocabs[n].items(), columns=['ngram', 'count'])
        df.to_csv(lang + '_' + affix + '_' + str(n) + '_vocab.csv')


def tag_sentences(model, sentence, language=None, mapping=None):
    """Part-of-speeh tag dataframe sentence."""
    ud = ''
    penn = ''
    if type(sentence) == str:
        doc = model(sentence)
        for token in doc:
            pos = token.pos_
            if token.pos_ == CONJ:
                pos = CCONJ if language and language == SPANISH else pos
            ud += pos + ' '
            tag = token.tag_
            if language == SPANISH and mapping:
                tag = mapping[tag]
            penn += tag + ' '
    return (ud, penn)


def get_structural_errors():
    """Read CLC error types file and return only
    the structural error types."""
    error_types = pd.read_csv('./data/error_type_meaning.csv')
    return error_types[error_types.structural == True]['error_type'].\
        tolist()


def time_since(since):
    """Compute passed time since the time parameter."""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def create_confusion_data(dataset_file, gold_column, guess_column):
    """Prepare confusion matrix data."""
    df = pd.read_csv(dataset_file)
    n_columns = 2
    confusion = [[0, 0], [0, 0]]
    for index, row in df.iterrows():
        if type(row[gold_column]) == bool and type(row[guess_column]) == bool:
            confusion[int(row[gold_column])][int(row[guess_column])] += 1
    for i in range(n_columns):
        total = sum(confusion[i])
        for j in range(n_columns):
            confusion[i][j] = confusion[i][j] / total
    return confusion


def power_of_ten_value(number):
    """Compute power of ten value equivalent to the parameter's size.
    E.g., power_of_ten_value(38) == 10, power_of_ten_value(422) == 100"""
    return int('1' + '0' * (len(str(int(number))) - 1))


def setup_train_test_data(dataset, percentage, gold_column):
    """Split dataset into train and test subsets."""
    df = pd.read_csv(dataset)
    y = df[gold_column]
    x_train, x_test, y_train, y_test = train_test_split(
                                        df, y, test_size=percentage)
    return x_train.copy(), x_test.copy()


def evaluate_models(filename, fields, l1, l2, model_label, gold_column):
    """Compare language models' probability results in the L1 and L2
    classifying the datapoints as language transfer or not, then compare
    this classification with the datapoints' gold labels."""
    df = pd.read_csv(filename)
    # If the probability in the L1 is greater than the probability in the L2
    # the sequence is tagged as negative language transfer
    df[model_label] = np.where(df[l1] > df[l2], True, False)
    df['result'] = np.where(df[model_label] == df[gold_column],
                            True, False)
    df = df[fields + [l1, l2, model_label, 'result']]
    df.to_csv(filename)
    print(filename)
    print(df.groupby(['result']).size().reset_index(name='count'))
