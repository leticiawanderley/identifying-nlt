import glob
import pandas as pd


def compute_accuracy(dataset_file):
    df = pd.read_csv(dataset_file, index_col=[0])
    lang, tagset, method, n = \
        dataset_file.split('/')[-1].replace('.csv', '').split('_')
    correct = 0
    for index, row in df.iterrows():
        result = ''
        if row['zhs'] > row['en']:
            result = 'zhs'
        elif row['en'] > row['zhs']:
            result = 'en'
        if result == lang:
            correct += 1
    f = open('n_gram_tuning.csv', 'a')
    f.write(lang + ',' + method + ',' + n + ',' + str(correct/df.shape[0]) + '\n')


def compute_all_files_accuracy():
    for file in glob.glob('./data/n-gram_tuning_results/*.csv'):
        compute_accuracy(file)


def group_accuracies(dataset_file):
    df = pd.read_csv(dataset_file, index_col=[0])
    grouped = df.groupby(['method', 'n']).mean()
    grouped.to_csv('./n_gram_tuning_grouped.csv')


group_accuracies('./n_gram_tuning.csv')
