import kenlm
import pandas as pd

from statistics import mean, median


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


def hyperparameter_tuning(n):
    arpa_template = \
        'data/training_data/chinese_english_splits/n-gram_tuning/{}_{}_{}.arpa'
    tuning_results = {'n': [], 'mean': [], 'median': []}
    for n in range(2, n + 1):
        tuning_results['n'].append(n)
        cv_results = []
        for f in range(1, 6):
            model_en = kenlm.LanguageModel(
                arpa_template.format('en', str(f), str(n)))
            model_zhs = kenlm.LanguageModel(
                arpa_template.format('zhs', str(f), str(n)))
            print('{0}-gram en model'.format(model_en.order))
            print('{0}-gram zhs model'.format(model_zhs.order))
            eval_dataset = \
                'data/training_data/chinese_english_splits/n-gram_tuning/test_fold_' + \
                str(f) + '.csv'
            cv_results.append(hyperparameter_eval(eval_dataset,
                              ['en', 'zhs'], model_en, model_zhs))
        tuning_results['mean'].append(mean(cv_results))
        tuning_results['median'].append(median(cv_results))
    pd.DataFrame.from_dict(tuning_results).to_csv(
        'data/tuning_results_n-gram/n-gram_tuning_results.csv')


hyperparameter_tuning(6)
