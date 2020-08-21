import os
import pandas as pd
import spacy

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
    df['type_and_trigram_ud'] = df['error_type'] + ' ' + \
        df['incorrect_trigram_ud']
    ud_quadgram, penn_quadgram = select_n_gram(df)
    df['ud_quadgram'] = ud_quadgram
    df['penn_quadgram'] = penn_quadgram
    df.to_csv(output_filename)


def select_n_gram(dataframe):
    nlp = spacy.load("en_core_web_lg")
    ud_n_grams = []
    penn_n_grams = []
    for index, row in dataframe.iterrows():
        ind = row['incorrect_error_index']
        sentence = row['incorrect_sentence']
        ind = ind - 1 if ind > 0 else ind
        sentence = sentence.replace('| ', '')
        doc = nlp(sentence)
        ud = []
        penn = []
        for i, token in enumerate(doc):
            ud.append(token.pos_)
            penn.append(token.tag_)
        ud_n_grams.append(' '.join(ud[ind:ind+4]))
        penn_n_grams.append(' '.join(penn[ind:ind+4]))
    return ud_n_grams, penn_n_grams


merge_datasets('data/testing data/annotated_FCE',
               ['error_type', 'Negative transfer?',
                'Likely reason for mistake',
                'correct_sentence', 'correct_trigram_poss',
                'incorrect_sentence', 'incorrect_trigram_poss',
                'incorrect_error_index'],
               'data/testing data/annotated_FCE/chinese_annotated_errors.csv')
