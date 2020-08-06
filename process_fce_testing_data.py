import os
import pandas as pd


def merge_datasets(folder, columns, output_filename):
    dataframes = []
    files = os.listdir(folder)
    for file in files:
        if 'FCE' in file:
            filename = folder + '/' + file
            dataframes.append(pd.read_csv(filename)[columns])
    df = pd.concat(dataframes)
    df = df[df['Negative transfer?'] == 'Y']
    df.to_csv(output_filename)


merge_datasets('data/testing data/annotated_FCE',
               ['error_type', 'Negative transfer?',
                'correct_sentence', 'correct_trigram_poss',
                'incorrect_sentence', 'incorrect_trigram_poss'],
               'chinese_annotated_errors.csv')
