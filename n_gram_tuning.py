import kenlm

import pandas as pd


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
