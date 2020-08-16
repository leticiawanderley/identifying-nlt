from constant import LEARNER_ENGLISH_FIELDS, ENGLISH, SPANISH,\
                     GOLD_LABEL, MODEL_LABEL
from utils import evaluate_n_gram_model


evaluate_n_gram_model(
    'data/results_learner_english_penn_trigram_unsmoothed.csv',
    LEARNER_ENGLISH_FIELDS, SPANISH, ENGLISH, MODEL_LABEL, GOLD_LABEL)
evaluate_n_gram_model(
    'data/results_learner_english_ud_trigram_unsmoothed.csv',
    LEARNER_ENGLISH_FIELDS, SPANISH, ENGLISH, MODEL_LABEL, GOLD_LABEL)

evaluate_n_gram_model(
    'data/results_learner_english_penn_trigram_laplace.csv',
    LEARNER_ENGLISH_FIELDS, SPANISH, ENGLISH, MODEL_LABEL, GOLD_LABEL)
evaluate_n_gram_model(
    'data/results_learner_english_ud_trigram_laplace.csv',
    LEARNER_ENGLISH_FIELDS, SPANISH, ENGLISH, MODEL_LABEL, GOLD_LABEL)

evaluate_n_gram_model(
    'data/results_learner_english_penn_trigram_interpolation.csv',
    LEARNER_ENGLISH_FIELDS, SPANISH, ENGLISH, MODEL_LABEL, GOLD_LABEL)
evaluate_n_gram_model(
    'data/results_learner_english_ud_trigram_interpolation.csv',
    LEARNER_ENGLISH_FIELDS, SPANISH, ENGLISH, MODEL_LABEL, GOLD_LABEL)
