import numpy as np
import pandas as pd

from constant import ANNOTATED_FCE_FIELDS


def evaluate_models(filename, fields, l1, l2):
    df = pd.read_csv(filename)
    # If the probability in the L1 is greater than the probability in the L2
    # the sequence is tagged as negative language transfer
    df['nlt'] = np.where(df[l1] > df[l2], True, False)
    results = []
    for index, row in df.iterrows():
        is_guess_correct = row['nlt'] == row['Negative transfer?']
        results.append(is_guess_correct)
    df['result'] = results
    df = df[fields + [l1, l2, 'nlt', 'result']]
    df.to_csv(filename)
    print(filename)
    print(df.groupby(['result']).size().reset_index(name='count'))


evaluate_models(
    'data/results_chinese_fce_incorrect_trigram_ud_unsmoothed.csv',
    ANNOTATED_FCE_FIELDS, 'zhs', 'en')

evaluate_models(
    'data/results_chinese_fce_incorrect_trigram_ud_laplace.csv',
    ANNOTATED_FCE_FIELDS, 'zhs', 'en')

evaluate_models(
    'data/results_chinese_fce_incorrect_trigram_ud_interpolation.csv',
    ANNOTATED_FCE_FIELDS, 'zhs', 'en')
