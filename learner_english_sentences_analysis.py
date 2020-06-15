import numpy as np
import pandas as pd


def evaluate_models(filename, fields):
    df = pd.read_csv(filename)
    # If the probability in English is greater than the probability in Spanish
    # the sentence is deemed correct (True)
    df['model_result'] = np.where(df['en'] > df['es'], True, False)
    df['model_result'] = np.where(df['model_result'] == df['correct'],
                                  True, False)
    df = df[fields + ['en', 'es', 'model_result']]
    df.to_csv(filename)
    print(df.groupby('model_result').count())


fields = ['sentence', 'correct', 'tags', 'trigram']
fields_poss = ['sentence', 'correct', 'tags', 'poss']
evaluate_models('data/results_learner_english_unsmoothed.csv',
                fields)
evaluate_models('data/results_learner_english_trigrams_unsmoothed.csv',
                fields)
evaluate_models('data/results_learner_english_unsmoothed_poss.csv',
                fields_poss)
evaluate_models('data/results_learner_english_laplace.csv',
                fields)
evaluate_models('data/results_learner_english_trigrams_laplace.csv',
                fields)
evaluate_models('data/results_learner_english_laplace_poss.csv',
                fields_poss)
evaluate_models('data/results_learner_english_interpolation.csv',
                fields)
evaluate_models('data/results_learner_english_trigrams_interpolation.csv',
                fields)
evaluate_models('data/results_learner_english_interpolation_poss.csv',
                fields_poss)
