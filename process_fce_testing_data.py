import os
import pandas as pd

from constant import ANNOTATED_FCE_FIELDS


def merge_datasets(folder, columns, output_filename):
    dataframes = []
    files = os.listdir(folder)
    for file in files:
        if 'FCE' in file:
            filename = folder + '/' + file
            dataframes.append(pd.read_csv(filename)[columns])
    df = pd.concat(dataframes)
    df = df[df['Likely reason for mistake'] != 'Omitted']
    df['Negative transfer?'] = df['Negative transfer?'] == 'Y'
    df.columns = ANNOTATED_FCE_FIELDS
    df.to_csv(output_filename)


merge_datasets('data/testing data/annotated_FCE',
               ['error_type', 'Negative transfer?',
                'Likely reason for mistake',
                'correct_sentence', 'correct_trigram_poss',
                'incorrect_sentence', 'incorrect_trigram_poss'],
               'data/testing data/annotated_FCE/chinese_annotated_errors.csv')
