import kenlm

import pandas as pd


def prepare_training_data(datasets, columns):
    for col in columns:
        output_file = open(col + '.txt', 'w')
        for dataset in datasets:
            df = pd.read_csv(dataset, index_col=[0])
            sequences = list(df[col])
            for s in sequences:
                output_file.write(s + '\n')
        output_file.close()


def test_fce(fce_errors, column):
    model_en = kenlm.LanguageModel(
        'data/training_data/chinese_english/en_5_full.arpa')
    model_zhs = kenlm.LanguageModel(
        'data/training_data/chinese_english/zhs_5_full.arpa')
    df = pd.read_csv(fce_errors, index_col=[0])
    errors = list(df[column])
    en = []
    zhs = []
    nlt = []
    for e in errors:
        en_score = 0
        zhs_score = 0
        if isinstance(e, str):
            en_score = model_en.score(e)
            zhs_score = model_zhs.score(e)
        en.append(en_score)
        zhs.append(zhs_score)
        nlt.append(zhs_score > en_score)
    df['en'] = en
    df['zhs'] = zhs
    df['nlt'] = nlt
    df.to_csv('data/results/kenlm_5_' + column + '.csv')


test_fce('data/test_data/zhs_structural_errors.csv',
         'incorrect_ud_tags_unigram')
test_fce('data/test_data/zhs_structural_errors.csv',
         'incorrect_ud_tags_bigram')
test_fce('data/test_data/zhs_structural_errors.csv',
         'incorrect_ud_tags_padded')
