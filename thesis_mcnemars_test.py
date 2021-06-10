import pandas as pd

from statsmodels.stats.contingency_tables import mcnemar


def compare_results(ngram_results, rnn_results):
    ngram = pd.read_csv(ngram_results, index_col=[0])
    ngram = list(ngram.apply(
                  lambda row: True if row['Negative transfer?'] == row['nlt'] else False,
                  axis=1))
    rnn = pd.read_csv(rnn_results, index_col=[0])
    rnn = list(rnn['result'])
    a, b, c, d = (0, 0, 0, 0)
    for i, value in enumerate(ngram):
        if value and rnn[i]:
            a += 1
        elif value:
            b += 1
        elif rnn[i]:
            c += 1
        else:
            d += 1
    return b/c, mcnemar([[a, b], [c, d]], exact=False, correction=True)


print(compare_results('data/results/kenlm_5_incorrect_ud_tags_padded.csv',
                      'data/results/results_en_ud_zhs_ud_0.0001_16_NLLoss_1_incorrect_ud_tags_padded.csv'))

print(compare_results('data/results/kenlm_5_incorrect_ud_tags_unigram.csv',
                      'data/results/results_en_ud_zhs_ud_0.0001_16_NLLoss_1_incorrect_ud_tags_unigram.csv'))


print(compare_results('data/results/kenlm_5_incorrect_ud_tags_bigram.csv',
                      'data/results/results_en_ud_zhs_ud_0.0001_16_NLLoss_1_incorrect_ud_tags_bigram.csv'))