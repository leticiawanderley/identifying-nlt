from constant import ANNOTATED_FCE_FIELDS, CHINESE, ENGLISH, \
                     GROUND_TRUTH, MODEL_LABEL
from utils import evaluate_n_gram_model
from visualization_functions import confusion_matrix

evaluate_n_gram_model(
    'data/results_chinese_fce_incorrect_ud_tags_bigram_interpolation.csv',
    ANNOTATED_FCE_FIELDS, CHINESE, ENGLISH,
    MODEL_LABEL, GROUND_TRUTH)

confusion_matrix(
    'data/results_chinese_fce_incorrect_ud_tags_bigram_interpolation.csv',
    GROUND_TRUTH, MODEL_LABEL,
    'confusion_matrix_zhs_en_incorrect_ud_tags_bigram_interpolation')
