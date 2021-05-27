import kenlm
import pandas as pd


def create_chinese_n_grams():
    n_grams = []
    for i in range(5):
        df = pd.read_csv('./data/training data/chinese-english_vocabs/zhs_ud_'
                         + str(i) + '_vocab.csv')
        n_grams += list(df['ngram'])
    return n_grams


def test():
    model_en = kenlm.LanguageModel('data/training data/en_5_full.arpa')
    model_zhs = kenlm.LanguageModel('data/training data/zhs_5_full.arpa')
    en = []
    zhs = []
    nlt = []
    n_grams = create_chinese_n_grams()
    for e in n_grams:
        en_score = 0
        zhs_score = 0
        if isinstance(e, str):
            en_score = model_en.score(e)
            zhs_score = model_zhs.score(e)
        en.append(en_score)
        zhs.append(zhs_score)
        nlt.append(zhs_score > en_score)
    df = pd.DataFrame()
    df['ngram'] = n_grams
    df['en'] = en
    df['zhs'] = zhs
    df['nlt'] = nlt
    df.to_csv('data/kenlm_5_chinese_ngrams.csv')


def check_fce_data():
    df = pd.read_csv('data/kenlm_5_chinese_ngrams.csv')
    df = df[df['nlt']==True]
    chinese = set(list(df['ngram']))
    fce = pd.read_csv('data/test_data/zhs_structural_errors.csv')
    #fce = fce[fce['Negative transfer?']==True]
    padded = list(fce['incorrect_ud_tags_padded'])
    unigram = list(fce['incorrect_ud_tags_unigram'])
    bigram = list(fce['incorrect_ud_tags_bigram'])
    count = 0
    for ngram in unigram:
        if ngram in chinese:
            count += 1
    print(count, len(unigram))


check_fce_data()
