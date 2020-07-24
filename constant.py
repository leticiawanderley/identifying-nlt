INTERPOLATION = 'interpolation'
LAPLACE = 'laplace'
UNSMOOTHED = 'unsmoothed'
OOV_TAG = '#'
NGRAM_METHODS = {UNSMOOTHED: [1, UNSMOOTHED],
                 LAPLACE: [3, LAPLACE],
                 INTERPOLATION: [3, INTERPOLATION]}
CONJ = 'CONJ'
CCONJ = 'CCONJ'
SPANISH = 'es'
PENN = {
    'en': {0: 'data/training data/en_penn_0_vocab.csv',
           1: 'data/training data/en_penn_1_vocab.csv',
           2: 'data/training data/en_penn_2_vocab.csv'},
    'es': {0: 'data/training data/es_penn_0_vocab.csv',
           1: 'data/training data/es_penn_1_vocab.csv',
           2: 'data/training data/es_penn_2_vocab.csv'}
}
UD_NGRAMS_FILES = {
    'en': {0: 'data/training data/en_ud_0_vocab.csv',
           1: 'data/training data/en_ud_1_vocab.csv',
           2: 'data/training data/en_ud_2_vocab.csv'},
    'es': {0: 'data/training data/es_ud_0_vocab.csv',
           1: 'data/training data/es_ud_1_vocab.csv',
           2: 'data/training data/es_ud_2_vocab.csv'}
}
LEARNER_ENGLISH_FIELDS = ['sentence', 'correct', 'error_type', 'penn',
                          'penn_trigram', 'ud', 'ud_trigram']
