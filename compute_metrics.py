import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score,\
                            recall_score, precision_score

from constant import MODEL_LABEL, GROUND_TRUTH
from utils import get_structural_errors


def dataset_metrics(datafile, output_file):
    metrics_dict = {
        'id': [],
        'total': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    df = pd.read_csv(datafile)
    df = df[df['error_type'].isin(get_structural_errors())]
    df = df[df[MODEL_LABEL].isin([True, False])]
    df = df[df[GROUND_TRUTH].isin([True, False])]
    metrics_dict['id'].append('overall')
    metrics_dict['total'].append(len(df.index))
    metrics_dict['precision'].append(
        round(precision_score(
            list(df[GROUND_TRUTH]),
            list(df[MODEL_LABEL])), 2))
    metrics_dict['recall'].append(
        round(recall_score(
            list(df[GROUND_TRUTH]),
            list(df[MODEL_LABEL])), 2))
    metrics_dict['f1'].append(
        round(f1_score(
            list(df[GROUND_TRUTH]),
            list(df[MODEL_LABEL])), 2))
    df['error_type'] = df.apply(lambda row: row['error_type'][0], axis=1)
    error_types = list(df.error_type.unique())
    print(confusion_matrix(list(df[GROUND_TRUTH]), list(df[MODEL_LABEL])))

    for t in error_types:
        data = df[df['error_type'] == t]
        metrics_dict['id'].append(t)
        metrics_dict['total'].append(len(data.index))
        metrics_dict['f1'].append(
            round(f1_score(
                list(data[GROUND_TRUTH]),
                list(data[MODEL_LABEL])), 2))
        metrics_dict['precision'].append(
            round(precision_score(
                list(data[GROUND_TRUTH]),
                list(data[MODEL_LABEL])), 2))
        metrics_dict['recall'].append(
            round(recall_score(
                list(data[GROUND_TRUTH]),
                list(data[MODEL_LABEL])), 2))
    pd.DataFrame.from_dict(metrics_dict).to_csv(output_file)


file = './data/results/kenlm_5_incorrect_ud_tags_padded.csv'
dataset_metrics(file, './data/results_metrics/metrics_kenlm_5_incorrect_ud_tags_padded.csv')

file = './data/results/kenlm_5_incorrect_ud_tags_unigram.csv'
dataset_metrics(file, './data/results_metrics/metrics_kenlm_5_incorrect_ud_tags_unigram.csv')

file = './data/results/kenlm_5_incorrect_ud_tags_bigram.csv'
dataset_metrics(file, './data/results_metrics/metrics_kenlm_5_incorrect_ud_tags_bigram.csv')
