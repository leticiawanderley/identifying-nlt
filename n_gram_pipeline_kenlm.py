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


def hyperparameter_eval(eval_dataset, languages, model_en, model_zhs):
    df = pd.read_csv(eval_dataset, index_col=[0])
    correct = 0
    for lang in languages:
        sequences = list(df[lang + '_ud'])
        for s in sequences:
            score_en = model_en.score(s)
            score_zhs = model_zhs.score(s)
            if lang == 'en' and score_en > score_zhs:
                correct += 1
            elif lang == 'zhs' and score_zhs > score_en:
                correct += 1
    return correct/(len(sequences) * 2)


def hyperparameter_tuning(eval_dataset, n):
    arpa_template = 'data/training_data/chinese_english/_splits/{}_{}_split.arpa'
    for i in range(2, n + 1):
        model_en = kenlm.LanguageModel(arpa_template.format('en', str(i)))
        model_zhs = kenlm.LanguageModel(arpa_template.format('zhs', str(i)))
        print('{0}-gram en model'.format(model_en.order))
        print('{0}-gram zhs model'.format(model_zhs.order))
        print(hyperparameter_eval(eval_dataset, ['en', 'zhs'], model_en,
                                  model_zhs))


def test_fce(fce_errors, column):
    model_en = kenlm.LanguageModel('data/training_data/chinese_english/en_5_full.arpa')
    model_zhs = kenlm.LanguageModel('data/training_data/chinese_english/zhs_5_full.arpa')
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

#hyperparameter_tuning('./data/training data/splits/eval_split.csv', 6)