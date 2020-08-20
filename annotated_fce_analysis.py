from constant import ANNOTATED_FCE_FIELDS, CHINESE, ENGLISH, \
                     GROUND_TRUTH, MODEL_LABEL
from utils import evaluate_n_gram_model
from visualization_functions import confusion_matrix


evaluate_n_gram_model(
    'data/results_chinese_fce_incorrect_trigram_ud_unsmoothed.csv',
    ANNOTATED_FCE_FIELDS, CHINESE, ENGLISH,
    MODEL_LABEL, GROUND_TRUTH)

evaluate_n_gram_model(
    'data/results_chinese_fce_incorrect_trigram_ud_laplace.csv',
    ANNOTATED_FCE_FIELDS, CHINESE, ENGLISH,
    MODEL_LABEL, GROUND_TRUTH)

evaluate_n_gram_model(
    'data/results_chinese_fce_incorrect_trigram_ud_interpolation.csv',
    ANNOTATED_FCE_FIELDS, CHINESE, ENGLISH,
    MODEL_LABEL, GROUND_TRUTH)


confusion_matrix(
    'data/results_chinese_fce_incorrect_trigram_ud_unsmoothed.csv',
    GROUND_TRUTH, MODEL_LABEL,
    'confusion_matrix_zhs_en_unsmoothed.png')

confusion_matrix(
    'data/results_chinese_fce_incorrect_trigram_ud_laplace.csv',
    GROUND_TRUTH, MODEL_LABEL,
    'confusion_matrix_zhs_en_laplace.png')

confusion_matrix(
    'data/results_chinese_fce_incorrect_trigram_ud_interpolation.csv',
    GROUND_TRUTH, MODEL_LABEL,
    'confusion_matrix_zhs_en_interpolation.png')
