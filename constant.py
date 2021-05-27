INTERPOLATION = 'interpolation'
LAPLACE = 'laplace'
UNSMOOTHED = 'unsmoothed'
OOV_TAG = '#'
NGRAM_METHODS = {UNSMOOTHED: [2, UNSMOOTHED],
                 LAPLACE: [2, LAPLACE],
                 INTERPOLATION: [2, INTERPOLATION]}
CONJ = 'CONJ'
CCONJ = 'CCONJ'
CHINESE = 'zhs'
ENGLISH = 'en'
SPANISH = 'es'
PENN_NGRAMS_FILES = {
    'en': {0: 'data/training data/europarl_vocabs/en_penn_0_vocab.csv',
           1: 'data/training data/europarl_vocabs/en_penn_1_vocab.csv',
           2: 'data/training data/europarl_vocabs/en_penn_2_vocab.csv'},
    'es': {0: 'data/training data/europarl_vocabs/es_penn_0_vocab.csv',
           1: 'data/training data/europarl_vocabs/es_penn_1_vocab.csv',
           2: 'data/training data/europarl_vocabs/es_penn_2_vocab.csv'}
}
UD_NGRAMS_FILES = {
    'en': {0: 'data/training data/europarl_vocabs/en_ud_0_vocab.csv',
           1: 'data/training data/europarl_vocabs/en_ud_1_vocab.csv',
           2: 'data/training data/europarl_vocabs/en_ud_2_vocab.csv'},
    'es': {0: 'data/training data/europarl_vocabs/es_ud_0_vocab.csv',
           1: 'data/training data/europarl_vocabs/es_ud_1_vocab.csv',
           2: 'data/training data/europarl_vocabs/es_ud_2_vocab.csv'}
}
UD_NGRAMS_FILES_ZH = {
    'en': {0: 'data/training data/chinese-english_vocabs/en_ud_0_vocab.csv',
           1: 'data/training data/chinese-english_vocabs/en_ud_1_vocab.csv',
           2: 'data/training data/chinese-english_vocabs/en_ud_2_vocab.csv'},
    'zhs': {0: 'data/training data/chinese-english_vocabs/zhs_ud_0_vocab.csv',
            1: 'data/training data/chinese-english_vocabs/zhs_ud_1_vocab.csv',
            2: 'data/training data/chinese-english_vocabs/zhs_ud_2_vocab.csv'}
}
UD_NGRAMS_FILES_SPLIT = {
    'en': {0: 'data/training data/splits/en_ud_0_vocab.csv',
           1: 'data/training data/splits/en_ud_1_vocab.csv',
           2: 'data/training data/splits/en_ud_2_vocab.csv',
           3: 'data/training data/splits/en_ud_3_vocab.csv',
           4: 'data/training data/splits/en_ud_4_vocab.csv'},
    'zhs': {0: 'data/training data/splits/zhs_ud_0_vocab.csv',
            1: 'data/training data/splits/zhs_ud_1_vocab.csv',
            2: 'data/training data/splits/zhs_ud_2_vocab.csv',
            3: 'data/training data/splits/zhs_ud_3_vocab.csv',
            4: 'data/training data/splits/zhs_ud_4_vocab.csv'}
}
LEARNER_ENGLISH_FIELDS = ['sentence', 'correct', 'error_type', 'penn',
                          'penn_trigram', 'ud', 'ud_trigram',
                          'Negative transfer?']
ANNOTATED_FCE_FIELDS = ['error_type', 'Negative transfer?',
                        'Likely reason for mistake',
                        'incorrect_ud_tags',
                        'incorrect_ud_tags_padded',
                        'incorrect_ud_tags_unigram',
                        'incorrect_ud_tags_bigram']
GROUND_TRUTH = 'Negative transfer?'
MODEL_LABEL = 'nlt'
CONFUSION_MATRIX_AXES = ['Not NLT', 'NLT']
DATA_FIELDS = ['student_id', 'raw_sentence',
               'overall_score', 'exam_score', 'error_type',
               'Negative transfer?', 'Likely reason for mistake',
               'error_length', 'incorrect_error_index',
               'incorrect_sentence']
