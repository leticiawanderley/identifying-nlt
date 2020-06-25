import numpy as np
import pandas as pd

from constant import LEARNER_ENGLISH_FIELDS


def evaluate_models(filename, fields):
    df = pd.read_csv(filename)
    # If the probability in English is greater than the probability in Spanish
    # the sentence is deemed correct (True)
    df['model_result'] = np.where(df['en'] > df['es'], True, False)
    df['model_result'] = np.where(df['model_result'] == df['correct'],
                                  True, False)
    df = df[fields + ['en', 'es', 'model_result']]
    df.to_csv(filename)
    print(filename)
    print(df.groupby(['model_result']).size().reset_index(name='count'))


evaluate_models('data/results_learner_english_tags_unsmoothed.csv',
                LEARNER_ENGLISH_FIELDS)
evaluate_models('data/results_learner_english_trigrams_unsmoothed.csv',
                LEARNER_ENGLISH_FIELDS)
evaluate_models('data/results_learner_english_poss_unsmoothed.csv',
                LEARNER_ENGLISH_FIELDS)
evaluate_models('data/results_learner_english_poss_trigrams_unsmoothed.csv',
                LEARNER_ENGLISH_FIELDS)

evaluate_models('data/results_learner_english_tags_laplace.csv',
                LEARNER_ENGLISH_FIELDS)
evaluate_models('data/results_learner_english_trigrams_laplace.csv',
                LEARNER_ENGLISH_FIELDS)
evaluate_models('data/results_learner_english_poss_laplace.csv',
                LEARNER_ENGLISH_FIELDS)
evaluate_models('data/results_learner_english_poss_trigrams_laplace.csv',
                LEARNER_ENGLISH_FIELDS)

evaluate_models('data/results_learner_english_tags_interpolation.csv',
                LEARNER_ENGLISH_FIELDS)
evaluate_models('data/results_learner_english_trigrams_interpolation.csv',
                LEARNER_ENGLISH_FIELDS)
evaluate_models('data/results_learner_english_poss_interpolation.csv',
                LEARNER_ENGLISH_FIELDS)
evaluate_models('data/results_learner_english_poss_trigrams_interpolation.csv',
                LEARNER_ENGLISH_FIELDS)