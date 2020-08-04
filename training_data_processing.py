import os
import pandas as pd

from utils import extract_vocabs, save_vocabs_to_csv, split_sentences


def list_training_datasets():
    """Create a list containing all training dataset files."""
    training_datasets = ['data/training data/tagged_sentences_1000sents.csv',
                         'data/training data/'
                         'tagged_sentences_dataset_sentences.csv']
    folder = 'data/training data/europarl'
    files = os.listdir(folder)
    for file in files:
        if 'tagged' in file:
            training_datasets.append(folder + '/' + file)
    return training_datasets


def pre_process_data(filenames, languages, column):
    """Format training data

    Concatenate values with a pipe separators between sentences.
    """
    datasets = {}
    for filename in filenames:
        df = pd.read_csv(filename)
        for language in languages:
            if language not in datasets.keys():
                datasets[language] = []
            datasets[language] += split_sentences(df[language + '_' + column].
                                                  tolist(), True)
    return datasets


def create_vocabs_files(n, languages):
    """Split training dataset into grams (from uni to ngrams)
    and count their occurences. Save resulting counts in csv files."""
    column = 'penn'
    training_files = list_training_datasets()
    datasets = pre_process_data(training_files, languages, column)
    for lang in languages:
        vocabs = extract_vocabs(datasets[lang], n)
        save_vocabs_to_csv(vocabs, lang, column)
