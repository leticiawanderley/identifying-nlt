INTERPOLATION = 'interpolation'
LAPLACE = 'laplace'
UNSMOOTHED = 'unsmoothed'
OOV_TAG = '#'
NGRAM_METHODS = {UNSMOOTHED: [1, UNSMOOTHED],
                 LAPLACE: [3, LAPLACE],
                 INTERPOLATION: [3, INTERPOLATION]}
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
UD_NGRAMS_FILES_GV = {
    'en': {0: 'data/training data/chinese-english_vocabs/en_ud_0_vocab.csv',
           1: 'data/training data/chinese-english_vocabs/en_ud_1_vocab.csv',
           2: 'data/training data/chinese-english_vocabs/en_ud_2_vocab.csv'},
    'zhs': {0: 'data/training data/chinese-english_vocabs/zhs_ud_0_vocab.csv',
            1: 'data/training data/chinese-english_vocabs/zhs_ud_1_vocab.csv',
            2: 'data/training data/chinese-english_vocabs/zhs_ud_2_vocab.csv'}
}
LEARNER_ENGLISH_FIELDS = ['sentence', 'correct', 'error_type', 'penn',
                          'penn_trigram', 'ud', 'ud_trigram',
                          'Negative transfer?']
ANNOTATED_FCE_FIELDS = ['error_type', 'Negative transfer?',
                        'Likely reason for mistake',
                        'correct_sentence', 'correct_trigram_ud',
                        'incorrect_sentence', 'incorrect_trigram_ud',
                        'incorrect_error_index']
GROUND_TRUTH = 'Negative transfer?'
MODEL_LABEL = 'nlt'
CONFUSION_MATRIX_AXES = ['Not NLT', 'NLT']
