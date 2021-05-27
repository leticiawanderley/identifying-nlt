import pandas as pd
import spacy

from constant import DATA_FIELDS, ANNOTATED_FCE_FIELDS


PIPE = '|'
SPACED_PIPE = ' | '
PIPE_AT_POS_0 = '| '
EMPTY = ''
SPACE = ' '


def create_feature_string(features):
    if not features:
        return ' '
    return ' '.join(features)


def read_data(filename):
    df = pd.read_csv(filename)
    df = df[DATA_FIELDS]
    return df


def create_ngrams(feature_list, index, length, error_type):
    raw_features = feature_list[index:index+length]
    padded_features = feature_list[index-1:index+length+1] \
        if index > 0 \
        else feature_list[index:index+length+1]
    unigram_features = feature_list[index:index+length+1]
    bigram_features = feature_list[index:index+length+2]
    return raw_features, padded_features, unigram_features, bigram_features


def extract_linguistic_features(nlp, sentence):
    if sentence.isupper():
        sentence = sentence.lower()
    tokens = []
    ptb_tags = []
    ud_tags = []
    deps = []
    index = None
    for sent in nlp.pipe([sentence], disable=["ner", "textcat"]):
        for i, token in enumerate(sent):
            if token.text == PIPE and index is None:
                index = i
            else:
                tokens.append(token.text)
                ptb_tags.append(token.tag_)
                ud_tags.append(token.pos_)
                deps.append(token.dep_)
    return {'ptb_tags': ptb_tags, 'ud_tags': ud_tags,
            'deps': deps, 'tokens': tokens}, index


def extract_ngrams(nlp, dataframe, column, length_column):
    ngrams_dict = {
        'tokens': [],
        'ptb_tags': [],
        'ud_tags': [],
        'deps': [],
        'tokens_padded': [],
        'ptb_tags_padded': [],
        'ud_tags_padded': [],
        'deps_padded': [],
        'tokens_unigram': [],
        'ptb_tags_unigram': [],
        'ud_tags_unigram': [],
        'deps_unigram': [],
        'tokens_bigram': [],
        'ptb_tags_bigram': [],
        'ud_tags_bigram': [],
        'deps_bigram': [],
    }
    for index, row in dataframe.iterrows():
        error_type = row['error_type']
        features_dict, index = \
            extract_linguistic_features(nlp, row[column])
        for key in features_dict:
            features, padded_features, unigram_features, bigram_features = \
                create_ngrams(features_dict[key], index,
                              int(row[length_column]), error_type)
            ngrams_dict[key].append(
                create_feature_string(features))
            ngrams_dict[key+'_padded'].append(
                create_feature_string(padded_features))
            ngrams_dict[key+'_unigram'].append(
                create_feature_string(unigram_features))
            ngrams_dict[key+'_bigram'].append(
                create_feature_string(bigram_features))
    return ngrams_dict


def process_fce_data(filename):
    df = read_data(filename)
    nlp = spacy.load('en_core_web_lg')
    incorrect_ngrams_dict = extract_ngrams(nlp, df, 'incorrect_sentence',
                                           'error_length')
    for key in incorrect_ngrams_dict:
        df['incorrect_'+key] = incorrect_ngrams_dict[key]
    df[ANNOTATED_FCE_FIELDS].to_csv('data/test_data/fce_processed_data.csv')


process_fce_data('data/test_data/nlt_dataset/main_chinese.csv')
