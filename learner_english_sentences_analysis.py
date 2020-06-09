import numpy as np
import pandas as pd


def evaluate_models(filename):
    df = pd.read_csv(filename)
    # If the probability in English is greater than the probability in Spanish
    # the sentence is deemed correct (True)
    df['model_result'] = np.where(df['en'] > df['es'], True, False)
    df['model_result'] = np.where(df['model_result'] == df['correct'],
                                  True, False)
    df.to_csv(filename)
    print(df.groupby('model_result').count())


evaluate_models('data/results_learner_english_unsmoothed.csv')
evaluate_models('data/results_learner_english_laplace.csv')
evaluate_models('data/results_learner_english_interpolation.csv')
