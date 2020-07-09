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
TAGS_NGRAMS_FILES = {
    'en': {0: 'data/training data/en_tags_0_vocab.csv',
           1: 'data/training data/en_tags_1_vocab.csv',
           2: 'data/training data/en_tags_2_vocab.csv'},
    'es': {0: 'data/training data/es_tags_0_vocab.csv',
           1: 'data/training data/es_tags_1_vocab.csv',
           2: 'data/training data/es_tags_2_vocab.csv'}
}
POSS_NGRAMS_FILES = {
    'en': {0: 'data/training data/en_poss_0_vocab.csv',
           1: 'data/training data/en_poss_1_vocab.csv',
           2: 'data/training data/en_poss_2_vocab.csv'},
    'es': {0: 'data/training data/es_poss_0_vocab.csv',
           1: 'data/training data/es_poss_1_vocab.csv',
           2: 'data/training data/es_poss_2_vocab.csv'}
}
LEARNER_ENGLISH_FIELDS = ['sentence', 'correct', 'error_type', 'tags',
                          'tags_trigram', 'poss', 'poss_trigram']